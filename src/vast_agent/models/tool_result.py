from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ErrorCode(StrEnum):
    SSH_TIMEOUT = "SSH_TIMEOUT"
    SSH_HOST_KEY_UNKNOWN = "SSH_HOST_KEY_UNKNOWN"
    SSH_HOST_KEY_CHANGED = "SSH_HOST_KEY_CHANGED"
    COMMAND_FAILED = "COMMAND_FAILED"
    PARSER_FAILED = "PARSER_FAILED"
    CONFIG_INVALID = "CONFIG_INVALID"
    HOST_NOT_FOUND = "HOST_NOT_FOUND"
    TOOL_UNSUPPORTED = "TOOL_UNSUPPORTED"


class ToolResult(BaseModel):
    success: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = Field(ge=0)
    timed_out: bool = False
    error_code: ErrorCode | None = None
    data: dict[str, Any] = Field(default_factory=dict)
