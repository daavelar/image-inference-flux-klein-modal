from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    aws_bucket: str
    aws_default_region: str = "auto"
    aws_endpoint: str
    aws_url: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_use_path_style_endpoint: bool = False

    loras_dir: Path = PROJECT_ROOT / "loras"
    models_base: Path | None = None
    require_network_volume: bool = False
    default_steps: int = 4
    inference_timeout: int = 300

    diffusion_model: str = "flux-2-klein-9b.safetensors"
    text_encoder: str = "qwen_3_8b_fp4mixed.safetensors"
    vae: str = "flux2-vae.safetensors"

    image_width: int = 1024
    image_height: int = 1024
    sampler_name: str = "euler"
    scheduler: str = "simple"
    cfg: float = 1.0
    flux_guidance: float = 4.0

    api_host: str = "0.0.0.0"
    api_port: int = 8002
    cost_per_second: float = 0.000542  # Modal L40S ($0.000542/s); override via COST_PER_SECOND

    comfy_input_dir: Path = PROJECT_ROOT / "vendor" / "comfyui" / "input"
    max_reference_images: int = 4
    max_reference_image_bytes: int = 10_485_760
    max_reference_image_max_side: int = 4096
    reference_download_timeout: int = 15


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.models_base is not None:
        base = settings.models_base.expanduser().resolve()
        return settings.model_copy(update={"loras_dir": base / "loras"})
    return settings
