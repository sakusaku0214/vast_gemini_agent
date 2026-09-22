from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

from vast_agent.agent.models import AgentResponse, FunctionCall


class GeminiClient(Protocol):
    calls: int
    def interact(self, *, model: str, inputs: Sequence[dict[str, object]],
                 system_instruction: str, tools: list[dict[str, object]],
                 thinking_level: str, store: bool) -> AgentResponse: ...


class GoogleInteractionsClient:
    """Thin adapter around google-genai's Interactions API (never generate_content)."""
    def __init__(self, api_key: str, api_version: str = "v1") -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key, http_options={"api_version": api_version})
        self.calls = 0

    def interact(self, *, model: str, inputs: Sequence[dict[str, object]],
                 system_instruction: str, tools: list[dict[str, object]],
                 thinking_level: str, store: bool) -> AgentResponse:
        self.calls += 1
        response = self._client.interactions.create(
            model=model, input=list(inputs), system_instruction=system_instruction,
            tools=tools, generation_config={"thinking_level": thinking_level}, store=store,
        )
        calls: list[FunctionCall] = []
        steps: list[dict[str, object]] = []
        for item in getattr(response, "steps", ()) or ():
            dumped = item.model_dump()
            steps.append(dumped)
            kind = getattr(item, "type", None) or dumped.get("type")
            if kind == "function_call":
                args = getattr(item, "arguments", None) or dumped.get("arguments", {})
                if isinstance(args, str): args = json.loads(args)
                calls.append(FunctionCall(name=getattr(item, "name", dumped.get("name")),
                                          arguments=args,
                                          call_id=getattr(item, "id", dumped.get("id", ""))))
        metadata = getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
        usage = {}
        aliases = {
            "input_tokens": ("input_tokens", "prompt_token_count"),
            "output_tokens": ("output_tokens", "candidates_token_count"),
            "thought_tokens": ("thought_tokens", "thoughts_token_count"),
            "cached_tokens": ("cached_tokens", "cached_content_token_count"),
            "tool_use_tokens": ("tool_use_tokens", "tool_use_prompt_token_count"),
            "total_tokens": ("total_tokens", "total_token_count"),
        }
        for target, names in aliases.items():
            usage[target] = next((getattr(metadata, n) for n in names
                                  if metadata is not None and getattr(metadata, n, None) is not None), None)
        return AgentResponse(output_text=getattr(response, "output_text", None), steps=steps,
                             function_calls=calls, usage=usage)


class ScriptedGeminiClient:
    def __init__(self, responses: Sequence[AgentResponse | Exception]) -> None:
        self.responses = list(responses); self.calls = 0; self.requests: list[dict[str, object]] = []

    def interact(self, **kwargs: object) -> AgentResponse:
        self.calls += 1; self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception): raise response
        response.function_calls = [
            FunctionCall(name=str(step["name"]), arguments=step.get("arguments", {}),
                         call_id=str(step.get("id", "")))
            for step in response.steps if step.get("type") == "function_call"
        ]
        return response
