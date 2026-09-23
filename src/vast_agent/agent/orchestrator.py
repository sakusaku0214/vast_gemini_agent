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
    """Small Gemini loop: think -> READ argv -> evidence -> think."""

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

    def investigate(self, _host: str | None, question: str) -> str:
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
        inputs: list[dict[str, object]] = [{
            "type": "text",
            "text": (
                "Configured hosts:\n"
                + json.dumps(enabled_hosts, ensure_ascii=False)
                + "\n\nHuman request:\n"
                + question
            ),
        }]

        while (
            llm_calls < self.settings.max_llm_calls
            and llm_calls < self.settings.max_agent_steps
            and time.monotonic() - start < self.settings.agent_wall_time_seconds
        ):
            if cancellation and cancellation.cancelled:
                return "調査はキャンセルされました。"

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
            inputs.extend(response.steps)

            if response.output_text and not response.function_calls:
                return response.output_text.strip()

            if not response.function_calls:
                break

            for call in response.function_calls:
                if cancellation and cancellation.cancelled:
                    return "調査はキャンセルされました。"
                if tool_calls >= self.settings.max_tool_calls:
                    inputs.append({
                        "type": "text",
                        "text": "READ上限に達しました。現在の証拠だけで回答してください。",
                    })
                    break

                output = self.functions.execute(
                    call.name,
                    call.arguments,
                    cancellation,
                )
                tool_calls += 1
                inputs.append({
                    "type": "function_result",
                    "name": call.name,
                    "call_id": call.call_id,
                    "result": [{
                        "type": "text",
                        "text": json.dumps(output, ensure_ascii=False, default=str),
                    }],
                })

        return "調査上限に達しました。得られた証拠だけでは回答を確定できませんでした。"
