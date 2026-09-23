from pydantic import BaseModel, ConfigDict, Field


class Capabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nvidia: bool = False
    docker: bool = False
    libvirt: bool = False
    vast: bool = False


class Host(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    vast_id: int | None = None
    address: str = Field(min_length=1)
    ssh_user: str = Field(min_length=1)
    ssh_port: int = Field(default=22, ge=1, le=65535)
    expected_gpu_count: int | None = Field(default=None, ge=1)
    aliases: list[str] = Field(default_factory=list)
    capabilities: Capabilities = Field(default_factory=Capabilities)
    enabled: bool = True
