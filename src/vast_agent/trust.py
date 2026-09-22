from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from vast_agent.models.host import Host


def trust_host(host: Host, known_hosts: Path, *, confirm: bool = True) -> tuple[bool, str]:
    target = f"[{host.address}]:{host.ssh_port}" if host.ssh_port != 22 else host.address
    scan = subprocess.run(
        ["ssh-keyscan", "-T", "10", "-p", str(host.ssh_port), host.address],
        capture_output=True, timeout=15, check=False,
    )
    keys = scan.stdout.decode("utf-8", errors="replace")
    if scan.returncode or not keys.strip(): return False, "ssh-keyscan did not return a host key"
    known_hosts.parent.mkdir(parents=True, exist_ok=True)
    known_hosts.touch(exist_ok=True)
    existing = subprocess.run(
        ["ssh-keygen", "-F", target, "-f", str(known_hosts)], capture_output=True, check=False,
    )
    if existing.returncode == 0:
        return False, "A key already exists for this host; remove it manually only after verifying a change"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(keys)
        temporary = Path(handle.name)
    try:
        fingerprints = subprocess.run(
            ["ssh-keygen", "-lf", str(temporary)], capture_output=True, check=False, text=True,
        ).stdout.strip()
    finally:
        temporary.unlink(missing_ok=True)
    message = f"Candidate keys for {host.name} ({target}):\n{fingerprints}"
    print(message)
    if confirm and input("Type YES after verifying these fingerprints: ").strip() != "YES":
        return False, "Registration cancelled; no key was written"
    with known_hosts.open("a", encoding="utf-8", newline="\n") as handle: handle.write(keys)
    return True, "Host keys registered"
