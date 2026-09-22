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


def test_normal_full_evidence_has_no_false_signatures():
    observation = build_observation("test-host", fixture_results("normal"))
    assert observation.gpu.nvml_observed and observation.gpu.pci_observed
    assert observation.system.health_observed
    assert observation.signatures == []


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


def test_pci_ignores_non_nvidia_vga_and_unrelated_audio():
    pci = parse_pci(fixture_results("mixed_vendors")["get_pci_status"].stdout)
    assert pci["pci_count"] == 1
    assert pci["nvidia_bound"] == 1
    assert [device.pci_address for device in pci["pci_devices"]] == [
        "0000:02:00.0", "0000:02:00.1",
    ]


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


def test_gpu_only_marks_pci_unobserved_without_false_signatures():
    results = fixture_results("normal")
    observation = build_observation("test-host", {
        "host_ping": results["host_ping"],
        "get_gpu_status": results["get_gpu_status"],
    })
    assert observation.gpu.nvml_observed and observation.gpu.nvml_ok
    assert not observation.gpu.pci_observed
    assert "NVML_UNAVAILABLE" not in observation.signatures
    assert "GPU_UNBOUND" not in observation.signatures


def test_pci_only_marks_nvml_unobserved_without_false_signature():
    results = fixture_results("normal")
    observation = build_observation("test-host", {
        "host_ping": results["host_ping"],
        "get_pci_status": results["get_pci_status"],
    })
    assert observation.gpu.pci_observed and observation.gpu.pci_ok
    assert not observation.gpu.nvml_observed
    assert observation.gpu.pci_count == 1
    assert "NVML_UNAVAILABLE" not in observation.signatures


def test_executed_gpu_and_pci_failures_produce_only_supported_signatures():
    results = fixture_results("normal")
    failed_gpu = results["get_gpu_status"].model_copy(
        update={"success": False, "stdout": "", "stderr": "nvidia-smi failed"},
    )
    assert "NVML_UNAVAILABLE" in build_observation("test-host", {
        "host_ping": results["host_ping"], "get_gpu_status": failed_gpu,
    }).signatures
    assert "GPU_UNBOUND" in build_observation("test-host", {
        "host_ping": results["host_ping"],
        "get_pci_status": fixture_results("unbound_gpu")["get_pci_status"],
    }).signatures


def test_garage_torrent_lspci_grep_output_handles_separator_and_vendors():
    from pathlib import Path

    pci = parse_pci(Path("tests/fixtures/garage_torrent_lspci.txt").read_text(encoding="utf-8"))
    assert pci["pci_count"] == 1
    assert pci["nvidia_bound"] == 1
    assert pci["unbound"] == 0
    assert [(device.device_type, device.driver) for device in pci["pci_devices"]] == [
        ("gpu", "nvidia"), ("audio", "snd_hda_intel"),
    ]


@pytest.mark.parametrize(
    ("gpu0_driver", "gpu1_driver", "expected"),
    [
        ("nvidia", "nvidia", (2, 2, 0, 0)),
        ("nvidia", "vfio-pci", (2, 1, 1, 0)),
        ("vfio-pci", "vfio-pci", (2, 0, 2, 0)),
        (None, "nvidia", (2, 1, 0, 1)),
    ],
)
def test_two_nvidia_gpu_binding_counts_exclude_audio_functions(
    gpu0_driver, gpu1_driver, expected,
):
    def driver_line(driver):
        return f"\n\tKernel driver in use: {driver}" if driver else ""

    output = (
        "0000:09:00.0 VGA compatible controller [0300]: NVIDIA Corporation Device [10de:2204]"
        f"{driver_line(gpu0_driver)}\n\tKernel modules: nouveau, nvidia\n"
        "0000:09:00.1 Audio device [0403]: NVIDIA Corporation Device [10de:1aef]\n"
        "\tKernel driver in use: snd_hda_intel\n\tKernel modules: snd_hda_intel\n"
        "0000:0a:00.0 3D controller [0302]: NVIDIA Corporation Device [10de:2235]"
        f"{driver_line(gpu1_driver)}\n\tKernel modules: nouveau, nvidia\n"
        "0000:0a:00.1 Audio device [0403]: NVIDIA Corporation Device [10de:1aef]\n"
        "\tKernel driver in use: vfio-pci\n\tKernel modules: snd_hda_intel\n"
    )

    pci = parse_pci(output)

    assert (
        pci["pci_count"], pci["nvidia_bound"], pci["vfio_bound"], pci["unbound"],
    ) == expected
    assert len([device for device in pci["pci_devices"] if device.device_type == "audio"]) == 2
