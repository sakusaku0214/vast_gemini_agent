import pytest
from conftest import fixture_results

from vast_agent.diagnostics.parser import build_observation


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
