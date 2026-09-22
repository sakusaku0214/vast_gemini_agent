import pytest
from conftest import fixture_results

from vast_agent.diagnostics.parser import build_observation, parse_pci
from vast_agent.execution.base import FakeExecutor
from vast_agent.inspector import inspect_host


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("normal", set()),
        ("fallen_off_bus", {"NVIDIA_FALLEN_OFF_BUS"}),
        ("nvml_failure", {"NVML_UNAVAILABLE"}),
        ("vfio_gpu", {"GPU_BOUND_VFIO", "NVML_UNAVAILABLE"}),
        ("ssh_failure", {"SSH_UNREACHABLE"}),
    ],
)
def test_fixture_raw_to_observation_to_signatures(case, expected):
    observation = build_observation("test-host", fixture_results(case))
    assert expected <= set(observation.signatures)


def test_gpu_is_structured():
    observation = build_observation("test-host", fixture_results("normal"))
    assert observation.gpu.nvml_ok
    assert observation.gpu.devices[0].uuid == "GPU-example"
    assert observation.gpu.devices[0].vram_total_mb == 24576


@pytest.mark.parametrize(
    ("case", "gpu_driver", "audio_driver"),
    [
        ("normal", "nvidia", "snd_hda_intel"),
        ("full_vfio", "vfio-pci", "vfio-pci"),
        ("mixed_binding", "nvidia", "vfio-pci"),
        ("unbound_gpu", None, "vfio-pci"),
    ],
)
def test_pci_gpu_and_audio_functions_are_structured(case, gpu_driver, audio_driver):
    devices = parse_pci(fixture_results(case)["get_pci_status"].stdout)["pci_devices"]
    assert [(device.device_type, device.driver) for device in devices] == [
        ("gpu", gpu_driver), ("audio", audio_driver),
    ]
    assert devices[0].function_group == devices[1].function_group == "0000:01:00"
    assert devices[0].modules


def test_registry_fixture_replay_reaches_observation_and_signatures(host):
    results = fixture_results("fallen_off_bus")
    observation = inspect_host(host, FakeExecutor(results))
    assert {"NVML_UNAVAILABLE", "NVIDIA_FALLEN_OFF_BUS"} <= set(observation.signatures)


@pytest.mark.parametrize(
    ("details", "signature"),
    [
        ("GPU has fallen off the bus", "NVIDIA_FALLEN_OFF_BUS"),
        ("NVRM: UVM fatal error", "NVIDIA_UVM_FATAL"),
        ("GSP RPC timeout", "NVIDIA_GSP_FAILURE"),
        ("GPU stuck in D3", "GPU_STUCK_D3"),
    ],
)
def test_precise_kernel_signatures(details, signature):
    results = fixture_results("normal")
    results["get_kernel_gpu_errors"] = results["get_kernel_gpu_errors"].model_copy(
        update={"stdout": details},
    )
    assert signature in build_observation("test-host", results).signatures


def test_binding_signatures_are_observed_but_vfio_is_not_implicitly_incident():
    assert "GPU_BOUND_VFIO" in build_observation(
        "test-host", fixture_results("full_vfio"),
    ).signatures
    assert "GPU_UNBOUND" in build_observation(
        "test-host", fixture_results("unbound_gpu"),
    ).signatures
