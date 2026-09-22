from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from vast_agent.config import ExternalToolsSettings
from vast_agent.execution.base import redact
from vast_agent.external_tools.http import ExternalProviderError, SafeJsonClient
from vast_agent.external_tools.models import FxArgs, ListCapabilitiesArgs, SearchArgs, WeatherArgs
from vast_agent.external_tools.providers import (
    BraveSearchProvider,
    FrankfurterFxProvider,
    OpenMeteoWeatherProvider,
    SearchProvider,
    UnavailableSearchProvider,
)


@dataclass(frozen=True)
class GeneralToolDefinition:
    name: str
    description: str
    category: str
    requires_network: bool
    requires_secret: bool
    argument_model: type[BaseModel]
    executor: Callable[[BaseModel], dict[str, object]]
    risk_class: str = "READ_ONLY"
    available: bool = True

    def declaration(self) -> dict[str, object]:
        return {"type": "function", "name": self.name, "description": self.description,
                "parameters": self.argument_model.model_json_schema()}


class GeneralToolRegistry:
    def __init__(self, definitions: list[GeneralToolDefinition], max_chars: int = 2500,
                 secrets: tuple[str, ...] = ()) -> None:
        self._tools = {item.name: item for item in definitions}
        self.max_chars, self.secrets = max_chars, secrets

    @property
    def declarations(self) -> list[dict[str, object]]:
        # Keep unavailable capabilities discoverable without inviting the model to call them.
        return [item.declaration() for item in self._tools.values() if item.available]

    def public_capabilities(self) -> list[dict[str, object]]:
        return [{"name": t.name, "description": t.description, "category": t.category,
                 "risk_class": t.risk_class, "requires_network": t.requires_network,
                 "available": t.available} for t in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": "FUNCTION_NOT_ALLOWED"}
        if not tool.available:
            return {"error": "FUNCTION_NOT_AVAILABLE", "function": name}
        try:
            args = tool.argument_model.model_validate(arguments)
            value = tool.executor(args)
        except ValidationError as exc:
            return {"error": "INVALID_ARGUMENTS", "details": exc.errors(include_input=False)}
        except ExternalProviderError:
            return {"error": "PROVIDER_UNAVAILABLE", "message": f"{name} is currently unavailable"}
        except Exception:
            return {"error": "PROVIDER_UNAVAILABLE", "message": f"{name} is currently unavailable"}
        compact = redact(json.dumps(value, ensure_ascii=False, default=str), self.secrets)
        if len(compact) > self.max_chars:
            compact = compact[:self.max_chars] + "…[truncated]"
        return {"untrusted_evidence": json.loads(compact) if not compact.endswith("[truncated]") else compact}


def build_general_registry(settings: ExternalToolsSettings, secrets: dict[str, str],
                           *, weather=None, fx=None, search: SearchProvider | None = None,
                           max_chars: int = 2500) -> GeneralToolRegistry:
    client = SafeJsonClient(
        {"geocoding-api.open-meteo.com", "api.open-meteo.com", "api.frankfurter.app",
         "api.search.brave.com"}, settings.request_timeout_seconds, settings.max_response_bytes,
    )
    weather = weather or OpenMeteoWeatherProvider(client)
    fx = fx or FrankfurterFxProvider(client)
    search = search or (BraveSearchProvider(client, secrets["SEARCH_API_KEY"])
                        if settings.search_provider == "brave" and secrets.get("SEARCH_API_KEY")
                        else UnavailableSearchProvider())
    registry: GeneralToolRegistry

    def weather_exec(value: BaseModel) -> dict[str, object]:
        args = WeatherArgs.model_validate(value)
        location = args.location or settings.default_weather_location
        if not location:
            return {"clarification_required": "どこの天気を確認しますか？"}
        return weather.get_weather(location, args.days)

    def list_exec(value: BaseModel) -> dict[str, object]:
        return {"available_capabilities": registry.public_capabilities(),
                "host_capabilities": "Host inspection and GPU/PCI/Docker/VM/Vast diagnostics",
                "write_boundary": "Approved operations require deterministic policy and OWNER approval"}

    definitions = [
        GeneralToolDefinition("get_weather", "Look up current weather and a short forecast for an explicitly supplied location.", "weather", True, False, WeatherArgs, weather_exec),
        GeneralToolDefinition("get_fx_rate", "Look up a current ISO currency exchange rate.", "finance", True, False, FxArgs, lambda value: fx.get_rate(FxArgs.model_validate(value).base_currency, FxArgs.model_validate(value).quote_currency)),
        GeneralToolDefinition("web_search", "Search bounded current public web information; this cannot fetch arbitrary URLs.", "search", True, settings.search_provider == "brave", SearchArgs, lambda value: search.search(SearchArgs.model_validate(value).query, SearchArgs.model_validate(value).max_results), available=search is not None and not isinstance(search, UnavailableSearchProvider)),
        GeneralToolDefinition("list_capabilities", "List currently registered capabilities and their safety boundaries.", "discovery", False, False, ListCapabilitiesArgs, list_exec),
    ]
    registry = GeneralToolRegistry(definitions, max_chars=max_chars, secrets=tuple(secrets.values()))
    return registry
