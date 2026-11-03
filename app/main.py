from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from app.comfy_runtime import initialize_comfyui_async, is_initialized, validate_storage_paths
from app.config import Settings, get_settings
from app.inference import InferenceError, generate_image, list_lora_files
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse, LoraInfo, LoraSpec

app = FastAPI(title="Flux Klein Inference API", version="1.0.0")


@app.on_event("startup")
async def startup() -> None:
    validate_storage_paths()
    await initialize_comfyui_async()


@app.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=is_initialized(),
        loras_count=len(list_lora_files(settings)),
    )


@app.get("/loras", response_model=list[LoraInfo])
def list_loras(settings: Settings = Depends(get_settings)) -> list[LoraInfo]:
    return [LoraInfo(filename=name) for name in list_lora_files(settings)]


@app.post("/generate", response_model=GenerateResponse)
def generate(
    request: GenerateRequest,
    settings: Settings = Depends(get_settings),
) -> GenerateResponse:
    try:
        return generate_image(
            settings,
            prompt=request.prompt,
            steps=request.steps,
            seed=request.seed,
            loras=[LoraSpec(filename=lora.filename, weight=lora.weight) for lora in request.loras],
            reference_images=request.reference_images or None,
            width=request.width,
            height=request.height,
            reference_megapixels=request.reference_megapixels,
        )
    except InferenceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
