from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WeatherArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    location: str | None = Field(default=None, max_length=200)
    days: int = Field(default=1, ge=1, le=3)

    @field_validator("location")
    @classmethod
    def valid_location(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or any(ord(c) < 32 for c in value):
            raise ValueError("location must be non-empty printable text")
        return value


class FxArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_currency: str = Field(min_length=3, max_length=3, pattern=r"^[A-Za-z]{3}$")
    quote_currency: str = Field(min_length=3, max_length=3, pattern=r"^[A-Za-z]{3}$")


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=300)
    max_results: int = Field(default=5, ge=1, le=5)


class ListCapabilitiesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
