from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

logger = logging.getLogger(__name__)

from app.comfy_runtime import ComfyUIError, initialize_comfyui, run_workflow
from app.config import Settings
from app.reference_images import ReferenceImageError, cleanup_reference_images, save_reference_images
from app.schemas import GenerateResponse, LoraSpec
from app.storage import upload_image
from app.workflow import build_workflow


class InferenceError(Exception):
    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def validate_loras(settings: Settings, loras: list[LoraSpec]) -> None:
    loras_dir = settings.loras_dir.resolve()
    for lora in loras:
        filename = Path(lora.filename).name
        if filename != lora.filename:
            raise InferenceError(f"Invalid lora filename: {lora.filename}", status_code=422)

        lora_path = (loras_dir / filename).resolve()
        if loras_dir not in lora_path.parents:
            raise InferenceError(f"Invalid lora path: {lora.filename}", status_code=422)
        if not lora_path.is_file():
            raise InferenceError(f"Lora not found: {filename}", status_code=422)


def list_lora_files(settings: Settings) -> list[str]:
    if not settings.loras_dir.is_dir():
        return []
    return sorted(
        path.name
        for path in settings.loras_dir.iterdir()
        if path.is_file() and path.suffix == ".safetensors"
    )


def generate_image(
    settings: Settings,
    *,
    prompt: str,
    steps: int | None = None,
    seed: int | None = None,
    loras: list[LoraSpec] | None = None,
    reference_images: list[str] | None = None,
    width: int | None = None,
    height: int | None = None,
    reference_megapixels: float | None = None,
) -> GenerateResponse:
    normalized_loras = [
        LoraSpec(filename=Path(lora.filename).name, weight=lora.weight)
        for lora in (loras or [])
    ]
    validate_loras(settings, normalized_loras)

    resolved_steps = steps if steps is not None else settings.default_steps
    resolved_seed = seed if seed is not None else random.randint(0, 2**63 - 1)
    resolved_width = width if width is not None else settings.image_width
    resolved_height = height if height is not None else settings.image_height

    logger.info(
        "[generate] %s",
        json.dumps(
            {
                "event": "started",
                "prompt": prompt,
                "steps": resolved_steps,
                "seed": resolved_seed,
                "width": resolved_width,
                "height": resolved_height,
                "loras": [
                    {"filename": lora.filename, "weight": lora.weight}
                    for lora in normalized_loras
                ],
                "reference_images": reference_images or [],
            },
            ensure_ascii=False,
        ),
    )

    reference_filenames: list[str] = []
    started_at = time.perf_counter()
    try:
        if reference_images:
            try:
                reference_filenames = save_reference_images(settings, reference_images)
            except ReferenceImageError as exc:
                raise InferenceError(str(exc), status_code=422) from exc

        workflow = build_workflow(
            settings=settings,
            prompt=prompt,
            steps=resolved_steps,
            seed=resolved_seed,
            loras=normalized_loras,
            reference_filenames=reference_filenames or None,
            width=resolved_width,
            height=resolved_height,
            reference_megapixels=reference_megapixels,
        )

        try:
            image_bytes = run_workflow(workflow)
        except ComfyUIError as exc:
            raise InferenceError(str(exc), status_code=502) from exc

        try:
            url, key = upload_image(settings, image_bytes)
        except Exception as exc:
            raise InferenceError(f"S3 upload failed: {exc}", status_code=500) from exc

        duration_seconds = round(time.perf_counter() - started_at, 2)
        response = GenerateResponse(
            url=url,
            key=key,
            seed=resolved_seed,
            duration_seconds=duration_seconds,
            cost=0.0,
        )
        logger.info(
            "[generate] %s",
            json.dumps(
                {
                    "event": "completed",
                    "prompt": prompt,
                    "steps": resolved_steps,
                    "seed": resolved_seed,
                    "width": resolved_width,
                    "height": resolved_height,
                    "loras": [
                        {"filename": lora.filename, "weight": lora.weight}
                        for lora in normalized_loras
                    ],
                    "reference_images": reference_images or [],
                    "published": bool(url),
                    "url": url,
                    "key": key,
                    "duration_seconds": duration_seconds,
                    "cost": response.cost,
                },
                ensure_ascii=False,
            ),
        )
        return response
    finally:
        if reference_filenames:
            cleanup_reference_images(settings, reference_filenames)


def generate_from_payload(settings: Settings, payload: dict) -> dict:
    """Normalize a JSON payload and run inference (used by Modal worker)."""
    if not isinstance(payload, dict):
        return {"error": "JSON body must be an object"}

    body = payload.get("input", payload)
    if not isinstance(body, dict):
        return {"error": "input must be an object"}

    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "prompt (non-empty string) required"}

    loras_raw = body.get("loras") or []
    if not isinstance(loras_raw, list):
        return {"error": "loras must be an array"}

    reference_images = body.get("reference_images") or []
    if not isinstance(reference_images, list):
        return {"error": "reference_images must be an array"}

    loras: list[LoraSpec] = []
    for item in loras_raw:
        if not isinstance(item, dict):
            return {"error": "each lora must be an object with filename and optional weight"}
        filename = item.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            return {"error": "each lora must include filename (string)"}
        weight = item.get("weight", 1.0)
        try:
            loras.append(LoraSpec(filename=filename.strip(), weight=float(weight)))
        except Exception as exc:
            return {"error": f"invalid lora entry: {exc}"}

    steps = body.get("steps")
    if steps is not None:
        try:
            steps = int(steps)
        except (TypeError, ValueError):
            return {"error": "steps must be an integer"}

    seed = body.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            return {"error": "seed must be an integer"}

    width = body.get("width")
    if width is not None:
        try:
            width = int(width)
        except (TypeError, ValueError):
            return {"error": "width must be an integer"}
        if width < 256 or width > 2048:
            return {"error": "width must be between 256 and 2048"}

    height = body.get("height")
    if height is not None:
        try:
            height = int(height)
        except (TypeError, ValueError):
            return {"error": "height must be an integer"}
        if height < 256 or height > 2048:
            return {"error": "height must be between 256 and 2048"}

    ref_strings: list[str] = []
    reference_megapixels = body.get("reference_megapixels")
    if reference_megapixels is not None:
        try:
            reference_megapixels = float(reference_megapixels)
        except (TypeError, ValueError):
            return {"error": "reference_megapixels must be a number"}
        if reference_megapixels <= 0 or reference_megapixels > 4:
            return {"error": "reference_megapixels must be > 0 and <= 4"}

    for item in reference_images:
        if not isinstance(item, str) or not item.strip():
            return {"error": "each reference_images entry must be a non-empty string"}
        ref_strings.append(item)

    if len(ref_strings) > settings.max_reference_images:
        return {
            "error": f"At most {settings.max_reference_images} reference images are allowed",
        }

    try:
        result = generate_image(
            settings,
            prompt=prompt.strip(),
            steps=steps,
            seed=seed,
            loras=loras,
            reference_images=ref_strings or None,
            width=width,
            height=height,
            reference_megapixels=reference_megapixels,
        )
        return result.model_dump()
    except InferenceError as exc:
        logger.warning(
            "[generate] %s",
            json.dumps(
                {
                    "event": "failed",
                    "prompt": prompt.strip(),
                    "steps": steps,
                    "seed": seed,
                    "error": str(exc),
                },
                ensure_ascii=False,
            ),
        )
        return {"error": str(exc)}
