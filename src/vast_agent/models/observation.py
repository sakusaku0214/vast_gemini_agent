from datetime import UTC, datetime

from pydantic import BaseModel, Field


class GpuDevice(BaseModel):
    index: int
    uuid: str | None = None
    model: str | None = None
    temperature_c: int | None = None
    utilization_percent: int | None = None
    vram_used_mb: int | None = None
    vram_total_mb: int | None = None
    power_state: str | None = None
    pci_bus: str | None = None


class PciDevice(BaseModel):
    pci_address: str
    device_type: str
    driver: str | None = None
    modules: list[str] = Field(default_factory=list)
    function_group: str


class GpuSummary(BaseModel):
    # Defaults keep observations written before observed-state tracking readable.
    nvml_observed: bool = False
    pci_observed: bool = False
    pci_ok: bool = False
    pci_count: int = 0
    nvml_ok: bool = False
    nvml_error: str | None = None
    nvidia_bound: int = 0
    vfio_bound: int = 0
    unbound: int = 0
    unknown_bound: int = 0
    devices: list[GpuDevice] = Field(default_factory=list)
    pci_devices: list[PciDevice] = Field(default_factory=list)

    @property
    def all_bound_to_vfio(self) -> bool:
        """Whether complete PCI evidence proves every physical GPU uses VFIO."""
        return (
            self.pci_observed
            and self.pci_ok
            and self.pci_count > 0
            and self.nvidia_bound == 0
            and self.vfio_bound == self.pci_count
            and self.unbound == 0
            and self.unknown_bound == 0
        )


class SystemSummary(BaseModel):
    health_observed: bool = False
    d_state_observed: bool = False
    d_state_processes: int = 0
    filesystem_max_percent: int | None = None
    failed_units: int = 0


class Observation(BaseModel):
    host: str
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ssh_ok: bool
    ssh_observed: bool = False
    gpu: GpuSummary = Field(default_factory=GpuSummary)
    system: SystemSummary = Field(default_factory=SystemSummary)
    services: dict[str, str] = Field(default_factory=dict)
    services_observed: list[str] = Field(default_factory=list)
    observed_tools: list[str] = Field(default_factory=list)
    signatures: list[str] = Field(default_factory=list)
    details: dict[str, object] = Field(default_factory=dict)
