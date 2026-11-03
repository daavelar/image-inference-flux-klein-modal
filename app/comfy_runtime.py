from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import uuid
from pathlib import Path

from app.config import PROJECT_ROOT
from app.logging_config import configure_aimdo_logging, configure_logging

COMFYUI_ROOT = PROJECT_ROOT / "vendor" / "comfyui"
MODELS_BASE = os.environ.get("MODELS_BASE", "").strip()

logger = logging.getLogger(__name__)

_inference_lock = threading.Lock()
_initialized = False
_executor = None


class ComfyUIError(Exception):
    pass


class _MockServer:
    # ComfyUI skips cached UI output when client_id is None (_send_cached_ui).
    client_id = "embedded"
    last_node_id = None
    last_prompt_id = None

    def send_sync(self, event, data, client_id):
        pass


def is_initialized() -> bool:
    return _initialized


def validate_storage_paths() -> None:
    from app.config import get_settings

    settings = get_settings()
    if not settings.require_network_volume:
        return

    volume_path = Path(os.environ.get("NETWORK_VOLUME_PATH", "/network"))
    if not volume_path.is_dir():
        raise RuntimeError(
            f"Network volume path not found at {volume_path}. "
            "Mount storage or adjust NETWORK_VOLUME_PATH."
        )

    models_base = settings.models_base or (volume_path / "models")
    required = (
        models_base / "diffusion_models",
        models_base / "text_encoders",
        models_base / "vae",
        settings.loras_dir,
    )
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise RuntimeError(
            "Required model directories not found on the network volume: "
            + ", ".join(missing)
        )

    print(f"[boot] models_base: {models_base}", flush=True)
    print(f"[boot] loras_dir: {settings.loras_dir}", flush=True)


async def initialize_comfyui_async() -> None:
    global _initialized, _executor

    if _initialized:
        return

    configure_logging()

    comfy_root = str(COMFYUI_ROOT)
    if comfy_root not in sys.path:
        sys.path.insert(0, comfy_root)

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")

    import comfy.options

    comfy.options.enable_args_parsing()

    old_argv = sys.argv
    sys.argv = [
        old_argv[0] if old_argv else "comfy_runtime",
        "--base-directory",
        comfy_root,
        "--disable-api-nodes",
        "--disable-all-custom-nodes",
    ]

    try:
        import comfy_aimdo.control
        from comfy.cli_args import enables_dynamic_vram
        import folder_paths
        import utils.extra_config

        extra_config_path = PROJECT_ROOT / "extra_model_paths.yaml"
        if MODELS_BASE:
            base = Path(MODELS_BASE)
            for sub in ("diffusion_models", "text_encoders", "vae", "loras"):
                folder_paths.add_model_folder_path(sub, str(base / sub))
        elif extra_config_path.is_file():
            utils.extra_config.load_extra_path_config(str(extra_config_path))

        folder_paths.set_output_directory(str(COMFYUI_ROOT / "output"))
        folder_paths.set_temp_directory(str(COMFYUI_ROOT / "temp"))
        os.makedirs(folder_paths.get_output_directory(), exist_ok=True)
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)

        if enables_dynamic_vram():
            comfy_aimdo.control.init()
            configure_aimdo_logging()

        import cuda_malloc  # noqa: F401
        import comfy.model_management
        import comfy.model_patcher
        import comfy.memory_management
        import execution
        import hook_breaker_ac10a0
        import nodes

        if (
            enables_dynamic_vram()
            and comfy.model_management.is_nvidia()
            and not comfy.model_management.is_wsl()
        ):
            device_index = comfy.model_management.get_torch_device().index
            if comfy_aimdo.control.init_device(device_index):
                comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
                comfy.memory_management.aimdo_enabled = True
                configure_aimdo_logging()
                logger.info("DynamicVRAM enabled")

        hook_breaker_ac10a0.save_functions()
        await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
        hook_breaker_ac10a0.restore_functions()

        _executor = execution.PromptExecutor(
            _MockServer(),
            cache_type=execution.CacheType.CLASSIC,
            cache_args={"lru": 0, "ram": 8.0},
        )
        _initialized = True
        logger.info("ComfyUI runtime initialized")
    finally:
        sys.argv = old_argv


def initialize_comfyui() -> None:
    asyncio.run(initialize_comfyui_async())


def run_workflow(workflow: dict) -> bytes:
    initialize_comfyui()

    with _inference_lock:
        import execution
        import folder_paths

        prompt_id = str(uuid.uuid4())
        _executor.reset()
        valid = asyncio.run(execution.validate_prompt(prompt_id, workflow, None))
        if not valid[0]:
            raise ComfyUIError(str(valid[1]))

        outputs_to_execute = valid[2]
        _executor.execute(workflow, prompt_id, {}, outputs_to_execute)

        if not _executor.success:
            raise ComfyUIError("Workflow execution failed")

        for node_output in _executor.history_result.get("outputs", {}).values():
            images = node_output.get("images", [])
            if not images:
                continue

            meta = images[0]
            output_dir = folder_paths.get_directory_by_type(meta.get("type", "output"))
            if output_dir is None:
                continue

            image_path = os.path.join(output_dir, meta.get("subfolder", ""), meta["filename"])
            with open(image_path, "rb") as image_file:
                return image_file.read()

        raise ComfyUIError("No image in workflow output")
