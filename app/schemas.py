from pydantic import BaseModel, Field


class LoraSpec(BaseModel):
    filename: str
    weight: float = Field(default=1.0, ge=0.0, le=2.0)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    steps: int | None = Field(default=None, ge=1, le=50)
    seed: int | None = None
    width: int | None = Field(default=None, ge=256, le=2048)
    height: int | None = Field(default=None, ge=256, le=2048)
    loras: list[LoraSpec] = Field(default_factory=list)
    reference_images: list[str] = Field(default_factory=list, max_length=4)
    reference_megapixels: float | None = Field(default=None, gt=0.0, le=4.0)
    callback_url: str | None = None
    external_id: int | str | None = None


class GenerateResponse(BaseModel):
    url: str
    key: str
    seed: int
    duration_seconds: float
    cost: float


class LoraInfo(BaseModel):
    filename: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    loras_count: int
