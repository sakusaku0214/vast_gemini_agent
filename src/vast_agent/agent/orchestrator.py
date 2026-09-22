from __future__ import annotations

import time

from vast_agent.agent.functions import FUNCTION_DECLARATIONS, FunctionExecutor
from vast_agent.agent.gemini import GeminiClient
from vast_agent.agent.models import InvestigationResult
from vast_agent.agent.prompts import SYSTEM_PROMPT
from vast_agent.config import GeminiSettings
from vast_agent.storage.database import Database


class InvestigationAgent:
    def __init__(self, client: GeminiClient, functions: FunctionExecutor, database: Database,
                 settings: GeminiSettings) -> None:
        self.client = client; self.functions = functions; self.database = database; self.settings = settings

    def investigate(self, host: str, question: str) -> InvestigationResult:
        start = time.monotonic(); tool_calls = 0; llm_calls = 0
        evidence: list[str] = []
        inputs: list[dict[str, object]] = [{"role": "user", "content":
            f"Target logical host: {host}\nInvestigation request: {question}"}]
        while (llm_calls < self.settings.max_llm_calls
               and llm_calls < self.settings.max_agent_steps
               and time.monotonic() - start < self.settings.agent_wall_time_seconds):
            try:
                response = self.client.interact(
                    model=self.settings.model, inputs=inputs, system_instruction=SYSTEM_PROMPT,
                    tools=FUNCTION_DECLARATIONS,
                    thinking_level=self.settings.investigate_thinking_level,
                    store=self.settings.store_interactions,
                )
            except Exception:  # API boundary; deliberately do not expose credential-bearing details
                return InvestigationResult(summary="Gemini unavailable. 取得済みObservationのみ表示します。",
                                           findings=evidence, missing_evidence=["Gemini analysis"])
            llm_calls += 1
            self.database.save_token_usage("investigate", self.settings.model,
                                           self.settings.investigate_thinking_level, response.usage)
            if response.text and not response.function_calls:
                return InvestigationResult(summary=response.text, findings=evidence,
                                           confidence="medium" if evidence else "low")
            if not response.function_calls:
                break
            inputs.append({"role": "assistant", "function_calls":
                           [call.model_dump() for call in response.function_calls]})
            results = []
            for call in response.function_calls:
                if tool_calls >= self.settings.max_tool_calls: break
                output = self.functions.execute(call.name, call.arguments, host)
                tool_calls += 1; evidence.append(f"{call.name}: {output}")
                results.append({"type": "function_result", "name": call.name,
                                "call_id": call.call_id, "result": output})
            inputs.append({"role": "tool", "content": results})
        return InvestigationResult(
            summary="調査上限に達しました。現在得られている証拠を返します。",
            findings=evidence, missing_evidence=["追加で必要な確認は次回調査で実施してください"],
        )
