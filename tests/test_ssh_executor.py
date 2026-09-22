from __future__ import annotations

import shlex
import subprocess
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from vast_agent.actions.executor import TypedActionExecutor
from vast_agent.actions.models import (
    ActionRequest,
    ActionType,
    ContainerParameters,
    GPUParameters,
    VMParameters,
)
from vast_agent.config import OperationsSettings
from vast_agent.execution.ssh import SSHExecutor
from vast_agent.tools.registry import TOOLS


def capture_remote_command(monkeypatch, tmp_path, host, command) -> tuple[str, list[str], Mock]:
    popen = Mock()
    popen.return_value.communicate.return_value = (b"ok", b"")
    popen.return_value.returncode = 0
    monkeypatch.setattr("vast_agent.execution.ssh.subprocess.Popen", popen)

    SSHExecutor(tmp_path / "known_hosts").execute(host, command, 20)

    ssh_argv = popen.call_args.args[0]
    return ssh_argv[-1], ssh_argv, popen


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        (
            "get_pci_status",
            "sh -c 'lspci -Dnnk | grep -A3 -Ei '\"'\"'VGA|3D|Audio'\"'\"''",
        ),
        (
            "get_system_health",
            "sh -c 'df -P; systemctl --failed --no-legend --plain'",
        ),
        (
            "get_docker_status",
            "sh -c 'systemctl is-active docker; docker ps -a --format '\"'\"'{{.ID}}|"
            "{{.Names}}|{{.Status}}|{{.Image}}'\"'\"''",
        ),
        (
            "get_vm_status",
            "sh -c 'virsh list --all; pgrep -a qemu-system || true'",
        ),
    ],
)
def test_registered_shell_tool_is_one_shell_safe_remote_command(
    monkeypatch, tmp_path, host, tool, expected,
):
    command = TOOLS[tool].command
    remote_command, ssh_argv, popen = capture_remote_command(
        monkeypatch, tmp_path, host, command,
    )

    assert remote_command == expected
    assert shlex.split(remote_command) == list(command)
    assert ssh_argv[-2] == f"{host.ssh_user}@{host.address}"
    assert popen.call_args.kwargs.get("shell") is not True


def test_garage_x570_pci_command_keeps_entire_script_as_dash_c_argument(
    monkeypatch, tmp_path, host,
):
    command = TOOLS["get_pci_status"].command
    remote_command, _, _ = capture_remote_command(monkeypatch, tmp_path, host, command)

    remote_argv = shlex.split(remote_command)
    assert remote_argv == [
        "sh", "-c", "lspci -Dnnk | grep -A3 -Ei 'VGA|3D|Audio'",
    ]
    assert len(remote_argv) == 3


@pytest.mark.parametrize(
    "tool",
    [
        "get_gpu_status",
        "get_gpu_processes",
        "get_kernel_gpu_errors",
        "get_vast_status",
        "get_vast_logs",
        "get_service_status",
        "get_journal_errors",
    ],
)
def test_plain_registered_argv_round_trips_through_remote_shell(
    monkeypatch, tmp_path, host, tool,
):
    command = TOOLS[tool].command
    remote_command, _, _ = capture_remote_command(monkeypatch, tmp_path, host, command)
    assert shlex.split(remote_command) == list(command)


def test_remote_shell_executes_serialized_argv_without_expanding_tokens():
    command = (
        "sh", "-c", 'printf "<%s>\\n" "$@"', "marker",
        "two words", "$(printf injected)", "semi;colon", "single'quote",
    )

    completed = subprocess.run(
        ["sh", "-c", shlex.join(command)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == [
        "<two words>", "<$(printf injected)>", "<semi;colon>", "<single'quote>",
    ]


def test_vm_script_validation_rejects_shell_metacharacters():
    with pytest.raises(ValidationError, match="shell metacharacters"):
        OperationsSettings(
            enable_vms_script="/opt/vm scripts/$(touch injected)/enable_vms.py",
        )


@pytest.mark.parametrize(
    ("action_request", "settings", "expected_argv"),
    [
        (
            ActionRequest(
                host="test-host",
                action_type=ActionType.RESTART_VAST_CONTAINER,
                parameters=ContainerParameters(container="C.123"),
            ),
            OperationsSettings(),
            ("sudo", "-n", "docker", "container", "restart", "C.123"),
        ),
        (
            ActionRequest(
                host="test-host",
                action_type=ActionType.GPU_RESET,
                parameters=GPUParameters(gpu_index=12),
            ),
            OperationsSettings(),
            ("sudo", "-n", "nvidia-smi", "--gpu-reset", "-i", "12"),
        ),
        (
            ActionRequest(
                host="test-host",
                action_type=ActionType.VM_MODE_ENABLE,
                parameters=VMParameters(mode="on"),
            ),
            OperationsSettings(
                enable_vms_script="/opt/vast tools/enable_vms.py",
            ),
            (
                "sudo", "-n", "python3",
                "/opt/vast tools/enable_vms.py", "on", "-f",
            ),
        ),
    ],
)
def test_typed_action_parameters_are_serialized_as_literal_tokens(
    monkeypatch, tmp_path, host, action_request, settings, expected_argv,
):
    action_argv = TypedActionExecutor(Mock(), settings).argv(action_request)
    remote_command, _, _ = capture_remote_command(monkeypatch, tmp_path, host, action_argv)

    assert tuple(shlex.split(remote_command)) == expected_argv
    assert remote_command == shlex.join(expected_argv)


def test_reboot_argv_is_shell_safe_and_has_no_shell_mode(monkeypatch, tmp_path, host):
    command = ("sudo", "-n", "systemctl", "reboot")
    remote_command, _, popen = capture_remote_command(monkeypatch, tmp_path, host, command)
    assert remote_command == "sudo -n systemctl reboot"
    assert popen.call_args.kwargs.get("shell") is not True
