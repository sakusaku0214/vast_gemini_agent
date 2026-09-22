from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from vast_agent.external_tools.http import ExternalProviderError, SafeJsonClient


class WeatherProvider(Protocol):
    def get_weather(self, location: str, days: int) -> dict[str, object]: ...


class FxProvider(Protocol):
    def get_rate(self, base: str, quote: str) -> dict[str, object]: ...


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int) -> dict[str, object]: ...


class OpenMeteoWeatherProvider:
    def __init__(self, client: SafeJsonClient) -> None:
        self.client = client

    def get_weather(self, location: str, days: int) -> dict[str, object]:
        geo = self.client.get("https://geocoding-api.open-meteo.com/v1/search", {
            "name": location, "count": 1, "language": "ja", "format": "json",
        })
        if not isinstance(geo, dict) or not isinstance(geo.get("results"), list) or not geo["results"]:
            raise ExternalProviderError("location not found")
        place = geo["results"][0]
        if not isinstance(place, dict) or not isinstance(place.get("latitude"), (int, float)):
            raise ExternalProviderError("provider returned malformed data")
        forecast = self.client.get("https://api.open-meteo.com/v1/forecast", {
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,precipitation,weather_code",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": days, "timezone": "auto",
        })
        if not isinstance(forecast, dict) or not isinstance(forecast.get("current"), dict):
            raise ExternalProviderError("provider returned malformed data")
        current = forecast["current"]
        daily = forecast.get("daily", {})
        return {
            "provider": "Open-Meteo", "resolved_location": place.get("name", location),
            "country": place.get("country"), "observed_at": current.get("time"),
            "temperature_c": current.get("temperature_2m"),
            "precipitation_mm": current.get("precipitation"),
            "weather_code": current.get("weather_code"),
            "forecast": daily,
        }


class FrankfurterFxProvider:
    _CURRENCIES = frozenset({
        "AUD", "BGN", "BRL", "CAD", "CHF", "CNY", "CZK", "DKK", "EUR", "GBP",
        "HKD", "HUF", "IDR", "ILS", "INR", "ISK", "JPY", "KRW", "MXN", "MYR",
        "NOK", "NZD", "PHP", "PLN", "RON", "SEK", "SGD", "THB", "TRY", "USD", "ZAR",
    })

    def __init__(self, client: SafeJsonClient) -> None:
        self.client = client

    def get_rate(self, base: str, quote: str) -> dict[str, object]:
        base, quote = base.upper(), quote.upper()
        if base not in self._CURRENCIES or quote not in self._CURRENCIES or base == quote:
            raise ExternalProviderError("unsupported currency pair")
        data = self.client.get("https://api.frankfurter.app/latest", {"from": base, "to": quote})
        if not isinstance(data, dict) or not isinstance(data.get("rates"), dict):
            raise ExternalProviderError("provider returned malformed data")
        rate = data["rates"].get(quote)
        if not isinstance(rate, (int, float)):
            raise ExternalProviderError("provider returned malformed data")
        return {"provider": "Frankfurter/ECB", "base": base, "quote": quote,
                "rate": rate, "provider_date": data.get("date")}


class UnavailableSearchProvider:
    def search(self, query: str, max_results: int) -> dict[str, object]:
        raise ExternalProviderError("web search is not configured")


class BraveSearchProvider:
    def __init__(self, client: SafeJsonClient, api_key: str) -> None:
        self.client, self.api_key = client, api_key

    def search(self, query: str, max_results: int) -> dict[str, object]:
        data = self.client.get("https://api.search.brave.com/res/v1/web/search",
                               {"q": query, "count": max_results},
                               {"Accept": "application/json", "X-Subscription-Token": self.api_key})
        web = data.get("web", {}) if isinstance(data, dict) else {}
        raw = web.get("results", []) if isinstance(web, dict) else []
        results = [{"title": item.get("title"), "url": item.get("url"),
                    "snippet": item.get("description"), "published": item.get("age")}
                   for item in raw[:max_results] if isinstance(item, dict)]
        return {"provider": "Brave Search", "searched_at": datetime.now(UTC).isoformat(),
                "query": query, "results": results}
