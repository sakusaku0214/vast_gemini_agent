from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
import urllib.request
from collections.abc import Iterable
from urllib.parse import urlencode, urlsplit, urlunsplit


class ExternalProviderError(RuntimeError):
    """A deliberately detail-free error safe to translate into a tool result."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


class SafeJsonClient:
    """Bounded HTTPS JSON transport restricted to explicit public provider hosts."""

    def __init__(self, allowed_hosts: Iterable[str], timeout: float, max_bytes: int) -> None:
        self.allowed_hosts = frozenset(host.casefold() for host in allowed_hosts)
        self.timeout = timeout
        self.max_bytes = max_bytes
        self._opener = urllib.request.build_opener(_NoRedirect)

    def get(self, base_url: str, params: dict[str, str | int],
            headers: dict[str, str] | None = None) -> object:
        parts = urlsplit(base_url)
        if (parts.scheme != "https" or not parts.hostname
                or parts.hostname.casefold() not in self.allowed_hosts
                or parts.username is not None or parts.password is not None):
            raise ExternalProviderError("provider URL is not allowed")
        self._assert_public(parts.hostname)
        url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), ""))
        request = urllib.request.Request(url, headers=headers or {}, method="GET")
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                if response.status != 200:
                    raise ExternalProviderError("provider unavailable")
                payload = response.read(self.max_bytes + 1)
        except (OSError, urllib.error.URLError, ValueError) as exc:
            raise ExternalProviderError("provider unavailable") from exc
        if len(payload) > self.max_bytes:
            raise ExternalProviderError("provider response too large")
        try:
            return json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExternalProviderError("provider returned malformed data") from exc

    @staticmethod
    def _assert_public(host: str) -> None:
        try:
            addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise ExternalProviderError("provider unavailable") from exc
        if not addresses:
            raise ExternalProviderError("provider unavailable")
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global:
                raise ExternalProviderError("provider address is not public")
