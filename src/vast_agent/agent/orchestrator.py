from __future__ import annotations

import json
import time

from pydantic import ValidationError

from vast_agent.actions.capability_bridge import ground_capability_gap
from vast_agent.agent.evidence import compact_evidence
from vast_agent.agent.functions import FunctionExecutor
from vast_agent.agent.gemini import GeminiClient
from vast_agent.agent.host_read import FUNCTION_DECLARATIONS, HOST_READ_CAPABILITIES
from vast_agent.agent.investigation_session import InvestigationSession, StopReason
from vast_agent.agent.models import InvestigationResult
from vast_agent.agent.prompts import GENERAL_SYSTEM_PROMPT, SYSTEM_PROMPT
from vast_agent.config import GeminiSettings
from vast_agent.external_tools.registry import GeneralToolRegistry
from vast_agent.jobs.cancellation import current_cancellation
from vast_agent.storage.database import Database


class InvestigationAgent:
    def __init__(self, client: GeminiClient, functions: FunctionExecutor, database: Database,
                 settings: GeminiSettings, general_tools: GeneralToolRegistry | None = None) -> None:
        self.client = client; self.functions = functions; self.database = database; self.settings = settings
        self.general_tools = general_tools

    def answer_general(self, question: str) -> str:
        """Run a bounded general loop exposing only the separate general READ registry."""
        start = time.monotonic(); tool_calls = 0; llm_calls = 0
        inputs: list[dict[str, object]] = [{"type": "text", "text": question}]
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
        session = InvestigationSession(
            target_host=host, goal=question,
            max_tool_calls=self.settings.max_tool_calls,
            max_rounds=min(self.settings.max_agent_steps, self.settings.max_llm_calls),
        )
        # Request-local state is retained only for diagnostics/tests; it grants no execution rights.
        self.last_session = session
        inputs: list[dict[str, object]] = [{
            "type": "text",
            "text": f"Target logical host: {host}\nInvestigation request: {question}",
        }]
        while (llm_calls < self.settings.max_llm_calls
               and llm_calls < self.settings.max_agent_steps
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
            for call in response.function_calls:
                if cancellation and cancellation.cancelled:
                    return InvestigationResult(summary="調査はキャンセルされました。", recommended_action="NONE",
                                               stop_reason=StopReason.CANCELLED)
                if not session.can_call():
                    return InvestigationResult(
                        summary="調査上限に達しました。現在得られている証拠を返します。",
                        findings=[item.summary for item in session.evidence],
                        missing_evidence=["未処理の追加確認"],
                        stop_reason=StopReason.BOUND_REACHED,
                    )
                if call.name not in HOST_READ_CAPABILITIES:
                    return InvestigationResult(
                        summary="未登録のREAD toolが選択されたため、安全に終了しました。",
                        findings=[item.summary for item in session.evidence],
                        missing_evidence=["registered READ tool"], recommended_action="NONE",
                        stop_reason=StopReason.ERROR,
                    )
                if session.has_call(call.name, call.arguments):
                    return InvestigationResult(
                        summary="同一のREADを再実行せず、取得済み証拠で終了しました。",
                        findings=[item.summary for item in session.evidence],
                        missing_evidence=["追加の異なる証拠"], recommended_action="NONE",
                        stop_reason=StopReason.NO_NEW_EVIDENCE,
                    )
                output = self.functions.execute(call.name, call.arguments, host)
                record = compact_evidence(call.name, output, max_chars=1800)
                if not session.add(call.name, call.arguments, record):
                    return InvestigationResult(
                        summary="証拠コンテキストの上限に達したため、安全に終了しました。",
                        findings=[item.summary for item in session.evidence],
                        missing_evidence=["上限を超える追加証拠"], recommended_action="NONE",
                        stop_reason=session.stop_reason or StopReason.BOUND_REACHED,
                    )
                inputs.append({
                    "type": "function_result", "name": call.name, "call_id": call.call_id,
                    "result": [{"type": "text", "text": json.dumps({
                        "untrusted_evidence_record": record.model_dump(mode="json"),
                    }, ensure_ascii=False)}],
                })
        return InvestigationResult(
            summary="調査上限に達しました。現在得られている証拠を返します。",
            findings=[item.summary for item in session.evidence],
            missing_evidence=["追加で必要な確認は次回調査で実施してください"],
            stop_reason=StopReason.BOUND_REACHED,
        )
