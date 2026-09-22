from __future__ import annotations

import json
import time

from pydantic import ValidationError

from vast_agent.agent.functions import FUNCTION_DECLARATIONS, FunctionExecutor
from vast_agent.agent.gemini import GeminiClient
from vast_agent.agent.models import InvestigationResult
from vast_agent.agent.prompts import SYSTEM_PROMPT
from vast_agent.config import GeminiSettings
from vast_agent.jobs.cancellation import current_cancellation
from vast_agent.storage.database import Database


class InvestigationAgent:
    def __init__(self, client: GeminiClient, functions: FunctionExecutor, database: Database,
                 settings: GeminiSettings) -> None:
        self.client = client; self.functions = functions; self.database = database; self.settings = settings

    def investigate(self, host: str, question: str) -> InvestigationResult:
        start = time.monotonic(); tool_calls = 0; llm_calls = 0
        cancellation = current_cancellation.get()
        evidence: list[str] = []
        inputs: list[dict[str, object]] = [{
            "type": "text",
            "text": f"Target logical host: {host}\nInvestigation request: {question}",
        }]
        while (llm_calls < self.settings.max_llm_calls
               and llm_calls < self.settings.max_agent_steps
               and time.monotonic() - start < self.settings.agent_wall_time_seconds):
            if cancellation and cancellation.cancelled:
                return InvestigationResult(summary="調査はキャンセルされました。", recommended_action="NONE")
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
            # With store=false, every model-generated step (including thought steps) must
            # be replayed verbatim into the next stateless interaction.
            inputs.extend(response.steps)
            if response.output_text and not response.function_calls:
                try:
                    return InvestigationResult.model_validate_json(response.output_text)
                except (ValidationError, ValueError):
                    return InvestigationResult(
                        summary="Geminiの構造化結論を検証できませんでした。",
                        findings=evidence,
                        recommended_action="NONE",
                        missing_evidence=["valid structured final response"],
                    )
            if not response.function_calls:
                break
            for call in response.function_calls:
                if cancellation and cancellation.cancelled:
                    return InvestigationResult(summary="調査はキャンセルされました。", recommended_action="NONE")
                if tool_calls >= self.settings.max_tool_calls:
                    return InvestigationResult(
                        summary="調査上限に達しました。現在得られている証拠を返します。",
                        findings=evidence,
                        missing_evidence=["未処理の追加確認"],
                    )
                output = self.functions.execute(call.name, call.arguments, host)
                tool_calls += 1; evidence.append(f"{call.name}: {output}")
                inputs.append({
                    "type": "function_result", "name": call.name, "call_id": call.call_id,
                    "result": [{"type": "text", "text": json.dumps(output, ensure_ascii=False)}],
                })
        return InvestigationResult(
            summary="調査上限に達しました。現在得られている証拠を返します。",
            findings=evidence, missing_evidence=["追加で必要な確認は次回調査で実施してください"],
        )
