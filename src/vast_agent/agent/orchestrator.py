from __future__ import annotations

import json
import logging
import time

from pydantic import ValidationError

from vast_agent.actions.capability_bridge import ground_capability_gap
from vast_agent.agent.evidence import compact_evidence
from vast_agent.agent.functions import FunctionExecutor
from vast_agent.agent.gemini import GeminiClient
from vast_agent.agent.host_read import FUNCTION_DECLARATIONS, HOST_READ_CAPABILITIES
from vast_agent.agent.investigation_session import InvestigationSession, StopReason
from vast_agent.agent.models import InvestigationResult
from vast_agent.agent.prompts import FINAL_SYNTHESIS_PROMPT, GENERAL_SYSTEM_PROMPT, SYSTEM_PROMPT
from vast_agent.config import GeminiSettings
from vast_agent.external_tools.registry import GeneralToolRegistry
from vast_agent.jobs.cancellation import current_cancellation
from vast_agent.storage.database import Database

logger = logging.getLogger(__name__)


def _user_input(text: str) -> dict[str, object]:
    """Build an explicit Interactions API user step suitable for stateless replay."""
    return {"type": "user_input", "content": [{"type": "text", "text": text}]}


class InvestigationAgent:
    def __init__(self, client: GeminiClient, functions: FunctionExecutor, database: Database,
                 settings: GeminiSettings, general_tools: GeneralToolRegistry | None = None) -> None:
        self.client = client; self.functions = functions; self.database = database; self.settings = settings
        self.general_tools = general_tools

    @staticmethod
    def _fallback_from_evidence(session: InvestigationSession, *, summary: str,
                                stop_reason: StopReason) -> InvestigationResult:
        return InvestigationResult(
            summary=summary,
            findings=[item.summary for item in session.evidence],
            missing_evidence=["valid final synthesis"],
            recommended_action="NONE",
            stop_reason=stop_reason,
        )

    def _synthesize_from_evidence(
        self,
        session: InvestigationSession,
        inputs: list[dict[str, object]],
        reason: StopReason,
    ) -> InvestigationResult:
        """Spend the reserved LLM call on a tool-free, evidence-only conclusion."""
        evidence = json.dumps(
            [item.model_dump(mode="json") for item in session.evidence], ensure_ascii=False,
        )
        synthesis_inputs = [*inputs, _user_input(FINAL_SYNTHESIS_PROMPT.format(
            reason=reason.value, evidence=evidence,
        ))]
        try:
            response = self.client.interact(
                model=self.settings.model, inputs=synthesis_inputs,
                system_instruction=SYSTEM_PROMPT, tools=[],
                thinking_level=self.settings.investigate_thinking_level,
                store=self.settings.store_interactions,
            )
        except Exception:  # API boundary; preserve evidence without leaking provider details
            logger.exception("Gemini final synthesis failed")
            session.stop_reason = StopReason.ERROR
            return self._fallback_from_evidence(
                session, summary="最終分析を完了できませんでした。取得済み証拠を返します。",
                stop_reason=StopReason.ERROR,
            )
        session.round += 1
        self.database.save_token_usage(
            "investigate_synthesis", self.settings.model,
            self.settings.investigate_thinking_level, response.usage,
        )
        # tools=[] is an execution boundary too: never honor an unexpected function call.
        if response.function_calls or not response.output_text:
            session.stop_reason = StopReason.BOUND_REACHED
            return self._fallback_from_evidence(
                session, summary="調査上限で最終分析を安全に完了できませんでした。取得済み証拠を返します。",
                stop_reason=StopReason.BOUND_REACHED,
            )
        try:
            result = InvestigationResult.model_validate_json(response.output_text)
        except (ValidationError, ValueError):
            session.stop_reason = StopReason.BOUND_REACHED
            return self._fallback_from_evidence(
                session, summary="Geminiの最終分析を検証できませんでした。取得済み証拠を返します。",
                stop_reason=StopReason.BOUND_REACHED,
            )
        result.capability_gaps = [
            ground_capability_gap(gap, session) for gap in result.capability_gaps
        ]
        if result.stop_reason in (None, StopReason.BOUND_REACHED, StopReason.NO_NEW_EVIDENCE):
            result.stop_reason = StopReason.ANSWERABLE
        session.stop_reason = StopReason(result.stop_reason)
        return result

    def answer_general(self, question: str) -> str:
        """Run a bounded general loop exposing only the separate general READ registry."""
        start = time.monotonic(); tool_calls = 0; llm_calls = 0
        inputs: list[dict[str, object]] = [_user_input(question)]
        declarations = self.general_tools.declarations if self.general_tools else []
        while (llm_calls < self.settings.max_llm_calls
               and llm_calls < self.settings.max_agent_steps
               and time.monotonic() - start < self.settings.agent_wall_time_seconds):
            try:
                response = self.client.interact(
                    model=self.settings.model, inputs=inputs,
                    system_instruction=GENERAL_SYSTEM_PROMPT, tools=declarations,
                    thinking_level=self.settings.default_thinking_level,
                    store=self.settings.store_interactions,
                )
            except Exception:  # API boundary; never expose provider or credential details
                logger.exception("Gemini general call failed")
                return "Gemini unavailable. 一般質問に回答できません。"
            llm_calls += 1
            self.database.save_token_usage(
                "general_tool" if response.function_calls else "general", self.settings.model,
                self.settings.default_thinking_level, response.usage,
            )
            inputs.extend(response.steps)
            if response.output_text and not response.function_calls:
                return response.output_text[:1200]
            if not response.function_calls:
                return "Gemini unavailable. 一般質問への回答を取得できませんでした。"
            for call in response.function_calls:
                if not self.general_tools:
                    return "この質問に利用できるREAD toolはありません。"
                if tool_calls >= self.settings.max_tool_calls:
                    return "一般READ toolの呼び出し上限に達したため、安全に終了しました。"
                output = self.general_tools.execute(call.name, call.arguments)
                tool_calls += 1
                inputs.append({
                    "type": "function_result", "name": call.name, "call_id": call.call_id,
                    "result": [{"type": "text", "text": json.dumps(
                        {"untrusted_tool_result": output}, ensure_ascii=False,
                    )}],
                })
        return "一般質問の処理上限に達したため、安全に終了しました。"

    def investigate(self, host: str, question: str) -> InvestigationResult:
        start = time.monotonic(); llm_calls = 0
        cancellation = current_cancellation.get()
        total_llm_budget = min(self.settings.max_agent_steps, self.settings.max_llm_calls)
        planning_llm_budget = max(0, total_llm_budget - 1)
        session = InvestigationSession(
            target_host=host, goal=question,
            max_tool_calls=self.settings.max_tool_calls,
            max_rounds=total_llm_budget,
        )
        # Request-local state is retained only for diagnostics/tests; it grants no execution rights.
        self.last_session = session
        inputs: list[dict[str, object]] = [_user_input(
            f"Target logical host: {host}\nInvestigation request: {question}"
        )]
        final_reason = StopReason.BOUND_REACHED
        while (llm_calls < planning_llm_budget
               and time.monotonic() - start < self.settings.agent_wall_time_seconds):
            if cancellation and cancellation.cancelled:
                return InvestigationResult(summary="調査はキャンセルされました。", recommended_action="NONE",
                                           stop_reason=StopReason.CANCELLED)
            try:
                response = self.client.interact(
                    model=self.settings.model, inputs=inputs, system_instruction=SYSTEM_PROMPT,
                    tools=FUNCTION_DECLARATIONS,
                    thinking_level=self.settings.investigate_thinking_level,
                    store=self.settings.store_interactions,
                )
            except Exception:  # API boundary; deliberately do not expose credential-bearing details
                logger.exception("Gemini investigate call failed")
                return InvestigationResult(summary="Gemini unavailable. 取得済みObservationのみ表示します。",
                                           findings=[item.summary for item in session.evidence],
                                           missing_evidence=["Gemini analysis"], stop_reason=StopReason.ERROR)
            llm_calls += 1
            session.round = llm_calls
            self.database.save_token_usage("investigate", self.settings.model,
                                           self.settings.investigate_thinking_level, response.usage)
            # With store=false, every model-generated step (including thought steps) must
            # be replayed verbatim into the next stateless interaction.
            inputs.extend(response.steps)
            if response.output_text and not response.function_calls:
                try:
                    result = InvestigationResult.model_validate_json(response.output_text)
                    result.capability_gaps = [
                        ground_capability_gap(gap, session) for gap in result.capability_gaps
                    ]
                    if result.stop_reason is None:
                        result.stop_reason = StopReason.ANSWERABLE
                    return result
                except (ValidationError, ValueError):
                    return InvestigationResult(
                        summary="Geminiの構造化結論を検証できませんでした。",
                        findings=[item.summary for item in session.evidence],
                        recommended_action="NONE",
                        missing_evidence=["valid structured final response"],
                        stop_reason=StopReason.ERROR,
                    )
            if not response.function_calls:
                break
            stop_execution = False
            for call_index, call in enumerate(response.function_calls):
                if cancellation and cancellation.cancelled:
                    return InvestigationResult(summary="調査はキャンセルされました。", recommended_action="NONE",
                                               stop_reason=StopReason.CANCELLED)
                if not session.can_call():
                    session.stop_reason = StopReason.BOUND_REACHED
                    final_reason = StopReason.BOUND_REACHED
                    self._append_unexecuted_results(
                        inputs, response.function_calls[call_index:], "READ_BOUND_REACHED",
                    )
                    stop_execution = True
                    break
                if call.name not in HOST_READ_CAPABILITIES:
                    return InvestigationResult(
                        summary="未登録のREAD toolが選択されたため、安全に終了しました。",
                        findings=[item.summary for item in session.evidence],
                        missing_evidence=["registered READ tool"], recommended_action="NONE",
                        stop_reason=StopReason.ERROR,
                    )
                if session.has_call(call.name, call.arguments):
                    session.stop_reason = StopReason.NO_NEW_EVIDENCE
                    final_reason = StopReason.NO_NEW_EVIDENCE
                    self._append_unexecuted_results(
                        inputs, response.function_calls[call_index:], "DUPLICATE_READ",
                    )
                    stop_execution = True
                    break
                output = self.functions.execute(call.name, call.arguments, host)
                record = compact_evidence(call.name, output, max_chars=1800)
                if not session.add(call.name, call.arguments, record):
                    final_reason = session.stop_reason or StopReason.BOUND_REACHED
                    self._append_executed_unretained_result(
                        inputs, call, "EVIDENCE_BOUND_REACHED",
                    )
                    self._append_unexecuted_results(
                        inputs, response.function_calls[call_index + 1:],
                        "EVIDENCE_BOUND_REACHED",
                    )
                    stop_execution = True
                    break
                inputs.append({
                    "type": "function_result", "name": call.name, "call_id": call.call_id,
                    "result": [{"type": "text", "text": json.dumps({
                        "untrusted_evidence_record": record.model_dump(mode="json"),
                    }, ensure_ascii=False)}],
                })
            if stop_execution:
                break
        if cancellation and cancellation.cancelled:
            return InvestigationResult(summary="調査はキャンセルされました。", recommended_action="NONE",
                                       stop_reason=StopReason.CANCELLED)
        if (total_llm_budget <= llm_calls
                or time.monotonic() - start >= self.settings.agent_wall_time_seconds):
            return self._fallback_from_evidence(
                session, summary="調査上限に達しました。現在得られている証拠を返します。",
                stop_reason=StopReason.BOUND_REACHED,
            )
        result = self._synthesize_from_evidence(session, inputs, final_reason)
        logger.info("Investigation complete trace=%s", session.trace())
        return result

    @staticmethod
    def _append_executed_unretained_result(
        inputs: list[dict[str, object]], call: object, reason: str,
    ) -> None:
        """Replay an executed READ without reintroducing its rejected raw result."""
        inputs.append({
            "type": "function_result", "name": call.name, "call_id": call.call_id,
            "result": [{"type": "text", "text": json.dumps({
                "read_executed": True, "result_retained": False, "reason": reason,
            })}],
        })

    @staticmethod
    def _append_unexecuted_results(
        inputs: list[dict[str, object]], calls: list[object], reason: str,
    ) -> None:
        """Complete replay protocol without executing or including rejected tool output."""
        for call in calls:
            inputs.append({
                "type": "function_result", "name": call.name, "call_id": call.call_id,
                "result": [{"type": "text", "text": json.dumps({
                    "read_executed": False, "reason": reason,
                })}],
            })
