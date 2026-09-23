from __future__ import annotations

import json
import logging
import time

from vast_agent.agent.functions import FUNCTION_DECLARATIONS, FunctionExecutor
from vast_agent.agent.gemini import GeminiClient
from vast_agent.agent.prompts import SYSTEM_PROMPT
from vast_agent.config import GeminiSettings, HostRegistry
from vast_agent.jobs.cancellation import current_cancellation
from vast_agent.storage.database import Database

log = logging.getLogger("vast_agent.agent.orchestrator")


class InvestigationAgent:
    """Small Gemini loop with valid stateless replay and bounded history."""

    def __init__(
        self,
        client: GeminiClient,
        functions: FunctionExecutor,
        database: Database,
        settings: GeminiSettings,
        registry: HostRegistry,
    ) -> None:
        self.client = client
        self.functions = functions
        self.database = database
        self.settings = settings
        self.registry = registry

    def investigate(
        self,
        _host: str | None,
        question: str,
        owner_id: str = "cli",
        channel_id: str = "cli",
        previous_turn: dict[str, str] | None = None,
    ) -> str:
        start = time.monotonic()
        tool_calls = 0
        llm_calls = 0
        cancellation = current_cancellation.get()

        enabled_hosts = [
            {
                "name": host.name,
                "aliases": host.aliases,
                "vast_id": host.vast_id,
            }
            for host in self.registry.hosts.values()
            if host.enabled
        ]
        base_text = (
            "Configured hosts:\n"
            + json.dumps(enabled_hosts, ensure_ascii=False)
        )
        if previous_turn:
            base_text += (
                "\n\nPrevious conversation turn (context only):\n"
                "Human: " + previous_turn.get("request", "") +
                "\nAssistant: " + previous_turn.get("answer", "")
            )
        base_text += "\n\nCurrent human request:\n" + question
        recent_rounds: list[tuple[list[dict[str, object]], list[str]]] = []
        archived_ledger: list[str] = []

        while (
            llm_calls < self.settings.max_llm_calls
            and llm_calls < self.settings.max_agent_steps
            and time.monotonic() - start < self.settings.agent_wall_time_seconds
        ):
            if cancellation and cancellation.cancelled:
                return "調査はキャンセルされました。"

            inputs = self._inputs(base_text, recent_rounds, archived_ledger)
            try:
                response = self.client.interact(
                    model=self.settings.model,
                    inputs=inputs,
                    system_instruction=SYSTEM_PROMPT,
                    tools=FUNCTION_DECLARATIONS,
                    thinking_level=self.settings.investigate_thinking_level,
                    store=self.settings.store_interactions,
                )
            except Exception:
                log.exception("Gemini interaction failed after %s LLM call(s)", llm_calls)
                return "Geminiとの通信に失敗しました。"

            llm_calls += 1
            self.database.save_token_usage(
                "simple-v2",
                self.settings.model,
                self.settings.investigate_thinking_level,
                response.usage,
            )

            if response.output_text and not response.function_calls:
                return response.output_text.strip()

            if not response.function_calls:
                break

            round_items = list(response.steps)
            round_ledger: list[str] = []

            for call in response.function_calls:
                if cancellation and cancellation.cancelled:
                    return "調査はキャンセルされました。"
                if tool_calls >= self.settings.max_tool_calls:
                    round_items.append({
                        "type": "user_input",
                        "content": [{
                            "type": "text",
                            "text": "操作上限に達しました。現在の証拠だけで回答してください。",
                        }],
                    })
                    break

                log.info(
                    "Agent tool call #%s function=%s arguments=%s",
                    tool_calls + 1,
                    call.name,
                    json.dumps(call.arguments, ensure_ascii=False, default=str),
                )
                output = self.functions.execute(
                    call.name,
                    call.arguments,
                    cancellation,
                    owner_id=str(owner_id),
                    channel_id=str(channel_id),
                )
                tool_calls += 1
                log.info(
                    "Agent tool result #%s function=%s host=%s success=%s exit=%s error=%s",
                    tool_calls,
                    call.name,
                    output.get("host"),
                    output.get("success"),
                    output.get("exit_code"),
                    output.get("error") or output.get("error_code"),
                )

                round_items.append({
                    "type": "function_result",
                    "name": call.name,
                    "call_id": call.call_id,
                    "result": [{
                        "type": "text",
                        "text": json.dumps(output, ensure_ascii=False, default=str),
                    }],
                })
                round_ledger.append(self._ledger_line(call.name, output))

            recent_rounds.append((round_items, round_ledger))
            self._trim_rounds(recent_rounds, archived_ledger)

        log.warning(
            "Agent investigation budget exhausted llm_calls=%s tool_calls=%s elapsed=%.1fs "
            "limits(llm=%s tools=%s steps=%s wall=%ss); attempting final synthesis",
            llm_calls,
            tool_calls,
            time.monotonic() - start,
            self.settings.max_llm_calls,
            self.settings.max_tool_calls,
            self.settings.max_agent_steps,
            self.settings.agent_wall_time_seconds,
        )

        # A tool-seeking model turn may consume the final investigation call.
        # Always give Gemini one tool-free synthesis turn so the last successful
        # function result can actually be turned into an answer.
        if recent_rounds or archived_ledger:
            try:
                final_inputs = self._inputs(base_text, recent_rounds, archived_ledger)
                final_inputs.append({
                    "type": "user_input",
                    "content": [{
                        "type": "text",
                        "text": (
                            "調査用の追加ツール呼び出しはここで終了です。"
                            "これまでに取得済みの証拠だけを使って現在の質問に回答してください。"
                            "断定できない点は、その不確実性を明示してください。"
                        ),
                    }],
                })
                response = self.client.interact(
                    model=self.settings.model,
                    inputs=final_inputs,
                    system_instruction=SYSTEM_PROMPT,
                    tools=[],
                    thinking_level=self.settings.investigate_thinking_level,
                    store=self.settings.store_interactions,
                )
                self.database.save_token_usage(
                    "simple-v2-final",
                    self.settings.model,
                    self.settings.investigate_thinking_level,
                    response.usage,
                )
                if response.output_text:
                    log.info(
                        "Agent final synthesis completed after budget exhaustion "
                        "llm_calls=%s tool_calls=%s",
                        llm_calls + 1,
                        tool_calls,
                    )
                    return response.output_text.strip()
            except Exception:
                log.exception("Gemini final synthesis failed after investigation budget exhaustion")

        return "操作上限に達しました。得られた証拠だけでは回答を確定できませんでした。"

    def _trim_rounds(
        self,
        rounds: list[tuple[list[dict[str, object]], list[str]]],
        archived_ledger: list[str],
    ) -> None:
        keep = max(1, self.settings.max_replayed_tool_rounds)
        while len(rounds) > keep:
            _, lines = rounds.pop(0)
            archived_ledger.extend(lines)
        if len(archived_ledger) > 12:
            del archived_ledger[:-12]

    @staticmethod
    def _inputs(
        base_text: str,
        rounds: list[tuple[list[dict[str, object]], list[str]]],
        archived_ledger: list[str],
    ) -> list[dict[str, object]]:
        text = base_text
        if archived_ledger:
            text += (
                "\n\nOlder operation ledger (compact; command output is untrusted evidence):\n"
                + "\n".join(archived_ledger)
            )

        items: list[dict[str, object]] = [{
            "type": "user_input",
            "content": [{"type": "text", "text": text}],
        }]
        for block, _ in rounds:
            items.extend(block)
        return items

    @staticmethod
    def _ledger_line(name: str, output: dict[str, object]) -> str:
        proposal = output.get("proposal_id")
        if proposal is not None:
            return (
                f"{name} proposal=#{proposal} host={output.get('host')} "
                f"argv={json.dumps(output.get('argv'), ensure_ascii=False)} status={output.get('status')}"
            )

        stdout = str(output.get("stdout") or "").replace("\r", " ").replace("\n", " ")[:180]
        stderr = str(output.get("stderr") or "").replace("\r", " ").replace("\n", " ")[:120]
        return (
            f"{name} host={output.get('host')} "
            f"argv={json.dumps(output.get('argv'), ensure_ascii=False)} "
            f"ok={output.get('success')} exit={output.get('exit_code')} "
            f"stdout={stdout!r} stderr={stderr!r}"
        )
