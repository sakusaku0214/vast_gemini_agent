from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from vast_agent.execution.ssh import openssh_path
from vast_agent.models.host import Host


def trust_host(host: Host, known_hosts: Path, *, confirm: bool = True) -> tuple[bool, str]:
    target = f"[{host.address}]:{host.ssh_port}" if host.ssh_port != 22 else host.address
    known_hosts.parent.mkdir(parents=True, exist_ok=True)
    known_hosts.touch(exist_ok=True)
    existing = subprocess.run(
        ["ssh-keygen", "-F", target, "-f", str(known_hosts)], capture_output=True, check=False,
    )
    if existing.returncode == 0:
        return False, "A key already exists for this host; remove it manually only after verifying a change"

    scan = subprocess.run(
        ["ssh-keyscan", "-T", "10", "-p", str(host.ssh_port), host.address],
        capture_output=True, timeout=15, check=False,
    )
    keys = scan.stdout.decode("utf-8", errors="replace")
    if (scan.returncode or not keys.strip()) and sys.platform == "win32":
        keys = _scan_with_windows_ssh(host)
    if not keys.strip():
        return False, "Neither ssh-keyscan nor the safe Windows SSH fallback returned a host key"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(keys)
        temporary = Path(handle.name)
    try:
        fingerprint_result = subprocess.run(
            ["ssh-keygen", "-lf", str(temporary)], capture_output=True, check=False, text=True,
        )
        fingerprints = fingerprint_result.stdout.strip()
    finally:
        temporary.unlink(missing_ok=True)
    if fingerprint_result.returncode or not fingerprints:
        return False, "The candidate host key could not be fingerprinted"
    message = f"Candidate keys for {host.name} ({target}):\n{fingerprints}"
    print(message)
    if confirm and input("Type YES after verifying these fingerprints: ").strip() != "YES":
        return False, "Registration cancelled; no key was written"
    with known_hosts.open("a", encoding="utf-8", newline="\n") as handle: handle.write(keys)
    return True, "Host keys registered"


def _scan_with_windows_ssh(host: Host) -> str:
    """Obtain a candidate via SSH without authentication or persistent trust changes."""
    with tempfile.TemporaryDirectory() as directory:
        candidate_file = Path(directory) / "candidate_known_hosts"
        candidate_file.touch()
        # accept-new applies only to this disposable file. The candidate is copied to the real
        # known_hosts only after its fingerprint is displayed and the operator types YES.
        subprocess.run(
            [
                "ssh", "-p", str(host.ssh_port),
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={openssh_path(candidate_file, windows=True)}",
                "-o", "GlobalKnownHostsFile=NUL",
                "-o", "PubkeyAuthentication=no",
                "-o", "PasswordAuthentication=no",
                "-o", "KbdInteractiveAuthentication=no",
                "-o", "PreferredAuthentications=none",
                "--", f"invalid-user@{host.address}", "exit",
            ],
            capture_output=True, timeout=15, check=False,
        )
        return candidate_file.read_text(encoding="utf-8", errors="replace")
