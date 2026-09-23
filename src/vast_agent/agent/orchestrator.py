from __future__ import annotations

import json
import time

from vast_agent.agent.functions import FUNCTION_DECLARATIONS, FunctionExecutor
from vast_agent.agent.gemini import GeminiClient
from vast_agent.agent.prompts import SYSTEM_PROMPT
from vast_agent.config import GeminiSettings, HostRegistry
from vast_agent.jobs.cancellation import current_cancellation
from vast_agent.storage.database import Database


class InvestigationAgent:
    """Small Gemini loop with a bounded replay window."""

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
        initial: dict[str, object] = {
            "type": "text",
            "text": (
                "Configured hosts:\n"
                + json.dumps(enabled_hosts, ensure_ascii=False)
                + "\n\nHuman request:\n"
                + question
            ),
        }
        recent_rounds: list[list[dict[str, object]]] = []
        ledger: list[str] = []

        while (
            llm_calls < self.settings.max_llm_calls
            and llm_calls < self.settings.max_agent_steps
            and time.monotonic() - start < self.settings.agent_wall_time_seconds
        ):
            if cancellation and cancellation.cancelled:
                return "調査はキャンセルされました。"

            inputs = self._inputs(initial, recent_rounds, ledger)
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
            for call in response.function_calls:
                if cancellation and cancellation.cancelled:
                    return "調査はキャンセルされました。"
                if tool_calls >= self.settings.max_tool_calls:
                    round_items.append({
                        "type": "text",
                        "text": "操作上限に達しました。現在の証拠だけで回答してください。",
                    })
                    break

                output = self.functions.execute(
                    call.name,
                    call.arguments,
                    cancellation,
                    owner_id=str(owner_id),
                    channel_id=str(channel_id),
                )
                tool_calls += 1
                round_items.append({
                    "type": "function_result",
                    "name": call.name,
                    "call_id": call.call_id,
                    "result": [{
                        "type": "text",
                        "text": json.dumps(output, ensure_ascii=False, default=str),
                    }],
                })
                ledger.append(self._ledger_line(call.name, output))

            recent_rounds.append(round_items)
            self._trim_rounds(recent_rounds)

        return "操作上限に達しました。得られた証拠だけでは回答を確定できませんでした。"

    def _trim_rounds(self, rounds: list[list[dict[str, object]]]) -> None:
        keep = max(1, self.settings.max_replayed_tool_rounds)
        if len(rounds) > keep:
            del rounds[:-keep]

    @staticmethod
    def _inputs(
        initial: dict[str, object],
        rounds: list[list[dict[str, object]]],
        ledger: list[str],
    ) -> list[dict[str, object]]:
        items = [initial]
        if ledger:
            items.append({
                "type": "text",
                "text": "Operation ledger (compact, untrusted outputs):\n" + "\n".join(ledger[-12:]),
            })
        for block in rounds:
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
