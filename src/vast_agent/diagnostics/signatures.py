from enum import StrEnum

from vast_agent.models.observation import Observation


class KnownSignature(StrEnum):
    SSH_UNREACHABLE = "SSH_UNREACHABLE"
    NVML_UNAVAILABLE = "NVML_UNAVAILABLE"
    GPU_UNBOUND = "GPU_UNBOUND"
    GPU_BOUND_VFIO = "GPU_BOUND_VFIO"
    GPU_STUCK_D3 = "GPU_STUCK_D3"
    NVIDIA_FALLEN_OFF_BUS = "NVIDIA_FALLEN_OFF_BUS"
    NVIDIA_UVM_FATAL = "NVIDIA_UVM_FATAL"
    NVIDIA_GSP_FAILURE = "NVIDIA_GSP_FAILURE"
    GPU_RESET_REQUIRED = "GPU_RESET_REQUIRED"
    KAALIA_ERROR = "KAALIA_ERROR"
    VAST_OFFLINE = "VAST_OFFLINE"
    DOCKER_FAILED = "DOCKER_FAILED"
    FILESYSTEM_HIGH_USAGE = "FILESYSTEM_HIGH_USAGE"
    FILESYSTEM_FULL = "FILESYSTEM_FULL"
    SYSTEMD_FAILED_UNIT = "SYSTEMD_FAILED_UNIT"


def detect_signatures(obs: Observation) -> list[KnownSignature]:
    found: list[KnownSignature] = []
    def add(condition: bool, signature: KnownSignature) -> None:
        if condition: found.append(signature)

    add(not obs.ssh_ok, KnownSignature.SSH_UNREACHABLE)
    add(obs.ssh_ok and obs.gpu.pci_count > 0 and not obs.gpu.nvml_ok, KnownSignature.NVML_UNAVAILABLE)
    add(obs.gpu.unbound > 0, KnownSignature.GPU_UNBOUND)
    add(obs.gpu.vfio_bound > 0, KnownSignature.GPU_BOUND_VFIO)
    text = "\n".join(str(value) for value in obs.details.values()).lower()
    add("fallen off the bus" in text, KnownSignature.NVIDIA_FALLEN_OFF_BUS)
    add("uvm" in text and any(x in text for x in ("fatal", "xid")), KnownSignature.NVIDIA_UVM_FATAL)
    add("gsp" in text and any(x in text for x in ("fail", "timeout")), KnownSignature.NVIDIA_GSP_FAILURE)
    add("d3cold" in text or "stuck in d3" in text, KnownSignature.GPU_STUCK_D3)
    add("gpu reset required" in text or "reset is required" in text, KnownSignature.GPU_RESET_REQUIRED)
    add("kaalia" in text and any(x in text for x in ("error", "failed")), KnownSignature.KAALIA_ERROR)
    add(obs.services.get("vastai") in {"inactive", "failed"}, KnownSignature.VAST_OFFLINE)
    add(obs.services.get("docker") in {"inactive", "failed"}, KnownSignature.DOCKER_FAILED)
    usage = obs.system.filesystem_max_percent or 0
    add(90 <= usage < 98, KnownSignature.FILESYSTEM_HIGH_USAGE)
    add(usage >= 98, KnownSignature.FILESYSTEM_FULL)
    add(obs.system.failed_units > 0, KnownSignature.SYSTEMD_FAILED_UNIT)
    return found
