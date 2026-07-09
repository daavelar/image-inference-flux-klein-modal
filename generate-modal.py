#!/usr/bin/env python3
"""
Flux Klein 9B image inference on Modal — HTTP + GPU worker.

Job payload (same shape as POST /generate):
    {"input": {"prompt", "steps"?, "seed"?, "loras"?, "reference_images"?}}
    Or top-level keys merged into input.

Storage: mount a Modal Volume (default ``fluxkleininf``) with models under
``<mount>/models/{diffusion_models,text_encoders,vae,loras}``.

Secrets (Modal Secret name from ``MODAL_AWS_SECRET``):
    AWS_BUCKET, AWS_ENDPOINT, AWS_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    Optional: AWS_DEFAULT_REGION, AWS_USE_PATH_STYLE_ENDPOINT

GPU: default fallback chain ``H200`` → ``H100`` → ``RTX-PRO-6000`` →
``A100-80GB`` → ``A100-40GB`` → ``L40S`` (override with ``MODAL_GPU``).

Cost: each job returns ``cost`` (USD) estimated from container wall time
(bootstrap, idle gaps, job runtime, amortized scaledown tail) using Modal list
prices for the detected GPU plus CPU/memory. Tune with ``MODAL_TAIL_IDLE_SECONDS``
(default 40% of ``scaledown_window``), ``MODAL_CPU_CORES``, ``MODAL_MEMORY_GIB``,
``MODAL_REGION_MULTIPLIER``.

Populate the volume once from your machine:
    python scripts/upload-models-to-modal-volume.py
    # or: modal volume put fluxkleininf ./models /models

Usage:
    modal deploy generate-modal.py
    modal run generate-modal.py --payload-json '{"input":{"prompt":"a portrait"}}'

HTTP (``flux_http_app`` — GET ``/health``, GET ``/loras``, POST ``/generate``):
    Default async: POST /generate returns 202 with ``call_id``; poll GET ``/result/{call_id}``.
    Sync: POST /generate?blocking=true

Docs: https://modal.com/docs/guide
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

VOL_NAME = os.environ.get("MODAL_VOLUME_NAME", "fluxkleininf")
VOL_MOUNT = os.environ.get("MODAL_VOLUME_MOUNT", "/my_vol")
_DEFAULT_GPU: list[str] = [
    "H200",
    "H100",
    "RTX-PRO-6000",
    "A100-80GB",
    "A100-40GB",
    "L40S",
]
MODAL_GPU_RAW = os.environ.get("MODAL_GPU", "").strip()
INFERENCE_GPU: str | list[str] = MODAL_GPU_RAW if MODAL_GPU_RAW else _DEFAULT_GPU

_MOUNT_IGNORE = [
    ".git/**/*",
    ".venv/**/*",
    "venv/**/*",
    "__pycache__/**/*",
    "*.py[cod]",
    ".cursor/**/*",
    ".claude/**/*",
    "diffusion_models/**/*",
    "text_encoders/**/*",
    "vae/**/*",
    "loras/**/*",
    "vendor/comfyui/output/**/*",
    "vendor/comfyui/input/**/*",
    "vendor/comfyui/temp/**/*",
]

_volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi>=0.115.0", "httpx>=0.27.0")
    .env({"PYTHONPATH": "/app"})
    .add_local_dir(str(HERE / "app"), remote_path="/app/app")
)

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime",
    )
    .env(
        {
            "DEBIAN_FRONTEND": "noninteractive",
            "PYTHONUNBUFFERED": "1",
            "PIP_NO_CACHE_DIR": "1",
            "PIP_BREAK_SYSTEM_PACKAGES": "1",
        }
    )
    .apt_install(
        "git",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
    )
    .workdir("/app")
    .add_local_dir(
        str(HERE),
        remote_path="/app",
        ignore=_MOUNT_IGNORE,
        copy=True,
    )
    .run_commands(
        'python -m pip install --upgrade pip && pip install -r "/app/requirements.txt"',
    )
)


def _configure_modal_env() -> None:
    vol = VOL_MOUNT
    models_base = str(Path(vol) / "models")
    os.environ["NETWORK_VOLUME_PATH"] = vol
    os.environ["MODELS_BASE"] = models_base
    os.environ["REQUIRE_NETWORK_VOLUME"] = "1"

    from app.config import get_settings

    get_settings.cache_clear()


def _optional_secrets() -> list[modal.Secret]:
    name = os.environ.get("MODAL_AWS_SECRET", "").strip()
    if not name:
        return []
    return [modal.Secret.from_name(name)]


def _job_from_http_body(body: object) -> dict:
    if not isinstance(body, dict):
        raise ValueError("JSON body must be an object")
    if "input" in body:
        return dict(body)
    return {"input": body}


def _split_job_and_callback(body: object) -> tuple[dict, str | None, int | str | None]:
    from app.callbacks import extract_callback_meta, validate_callback_url

    if not isinstance(body, dict):
        raise ValueError("JSON body must be an object")

    if "input" in body and isinstance(body["input"], dict):
        outer = dict(body)
        inner = dict(outer["input"])
        callback_url = outer.pop("callback_url", None)
        if callback_url is None:
            callback_url = inner.pop("callback_url", None)
        external_id = outer.pop("external_id", None)
        if external_id is None:
            external_id = inner.pop("external_id", None)
        outer["input"] = inner
        if callback_url is not None:
            callback_url = validate_callback_url(str(callback_url))
        if external_id is not None and not isinstance(external_id, (int, str)):
            raise ValueError("external_id must be an integer or string")
        return _job_from_http_body(outer), callback_url, external_id

    job_body, callback_url, external_id = extract_callback_meta(dict(body))
    return _job_from_http_body(job_body), callback_url, external_id


def _list_loras_from_volume() -> list[dict]:
    loras_dir = Path(VOL_MOUNT) / "models" / "loras"
    if not loras_dir.is_dir():
        return []
    return [
        {"filename": path.name}
        for path in sorted(loras_dir.iterdir())
        if path.is_file() and path.suffix == ".safetensors"
    ]


def _build_fastapi_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, Response

    api = FastAPI(title="Flux Klein Inference", version="1.0")

    @api.get("/health")
    def health_get():
        loras_count = len(_list_loras_from_volume())
        models_dir = Path(VOL_MOUNT) / "models"
        model_ready = (models_dir / "diffusion_models").is_dir()
        return JSONResponse(
            {
                "status": "ok",
                "model_ready": model_ready,
                "loras_count": loras_count,
            }
        )

    @api.head("/health")
    def health_head():
        return Response(status_code=200)

    @api.get("/loras")
    def list_loras():
        return JSONResponse(_list_loras_from_volume())

    @api.get("/result/{call_id}")
    def get_job_result(call_id: str):
        fc = modal.FunctionCall.from_id(call_id)
        try:
            result = fc.get(timeout=0)
        except TimeoutError:
            return JSONResponse(
                {"status": "pending", "call_id": call_id},
                status_code=202,
            )
        except Exception as e:
            return JSONResponse(
                {"status": "failed", "call_id": call_id, "error": str(e)},
                status_code=500,
            )
        if result.get("error"):
            payload = dict(result)
            payload.setdefault("status", "failed")
            payload.setdefault("call_id", call_id)
            return JSONResponse(payload, status_code=400)
        payload = dict(result)
        payload.setdefault("status", "completed")
        payload.setdefault("call_id", call_id)
        return JSONResponse(payload, status_code=200)

    @api.post("/generate")
    def generate_http(body: dict, blocking: bool = False):
        from app.callbacks import build_callback_payload, post_callback

        try:
            job, callback_url, external_id = _split_job_and_callback(body)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        worker = FluxKleinWorker()

        if blocking:
            try:
                result = worker.run_job.remote(job)
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
            if callback_url:
                payload = build_callback_payload("blocking", result, external_id)
                post_callback(callback_url, payload)
            status_code = 400 if result.get("error") else 200
            return JSONResponse(result, status_code=status_code)

        try:
            fc = worker.run_job.spawn(job)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        cid = fc.object_id
        if callback_url:
            deliver_callback.spawn(cid, callback_url, external_id)
        return JSONResponse(
            {
                "status": "pending",
                "call_id": cid,
                "task_id": cid,
                "result_url": f"/result/{cid}",
            },
            status_code=202,
        )

    return api


app = modal.App("flux-klein-modal")


@app.cls(
    image=image,
    gpu=INFERENCE_GPU,
    volumes={VOL_MOUNT: _volume},
    secrets=_optional_secrets(),
    timeout=60 * 30,
    scaledown_window=300,
)
class FluxKleinWorker:
    @modal.enter()
    def bootstrap(self):
        from app.modal_billing import ContainerCostTracker, detect_gpu_cost_per_second, get_gpu_name

        self._cost_tracker = ContainerCostTracker.from_env(scaledown_window=300)
        boot_started = self._cost_tracker.on_boot_start()

        _configure_modal_env()
        from app.comfy_runtime import initialize_comfyui, validate_storage_paths
        from app.config import get_settings

        gpu_name = get_gpu_name()
        cost_per_second = detect_gpu_cost_per_second()
        os.environ["COST_PER_SECOND"] = str(cost_per_second)
        get_settings.cache_clear()

        validate_storage_paths()
        print(
            f"[boot] gpu={gpu_name or 'unknown'} cost_per_second={cost_per_second}",
            flush=True,
        )
        print("[boot] loading ComfyUI runtime...", flush=True)
        initialize_comfyui()
        self.settings = get_settings()
        self._cost_tracker.on_boot_complete(boot_started)
        print(
            f"[boot] ready bootstrap_seconds={self._cost_tracker._bootstrap_seconds:.1f}",
            flush=True,
        )

    @modal.method()
    def run_job(self, job: dict) -> dict:
        from app.inference import generate_from_payload
        from app.modal_billing import run_with_billing

        return run_with_billing(
            self._cost_tracker,
            generate_from_payload,
            self.settings,
            job,
        )


@app.function(
    image=web_image,
    volumes={VOL_MOUNT: _volume},
    secrets=_optional_secrets(),
    timeout=60 * 30,
    scaledown_window=300,
)
def deliver_callback(call_id: str, callback_url: str, external_id: int | str | None = None):
    from app.callbacks import deliver_callback_for_call

    deliver_callback_for_call(call_id, callback_url, external_id)


@app.function(
    image=web_image,
    volumes={VOL_MOUNT: _volume},
    timeout=60 * 30,
    scaledown_window=300,
)
@modal.asgi_app(label="flux-http")
def flux_http_app():
    return _build_fastapi_app()


@app.local_entrypoint()
def main(payload_json: str | None = None):
    worker = FluxKleinWorker()
    raw = payload_json or os.environ.get("MODAL_JOB_JSON", "").strip()
    if not raw:
        raise SystemExit(
            "Set MODAL_JOB_JSON or run with --payload-json, e.g. "
            '\'{"input":{"prompt":"a portrait of a woman, studio lighting"}}\''
        )
    job = json.loads(raw)
    out = worker.run_job.remote(job)
    print(json.dumps(out, indent=2, default=str))
