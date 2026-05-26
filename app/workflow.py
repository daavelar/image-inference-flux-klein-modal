from __future__ import annotations

from app.config import Settings
from app.schemas import LoraSpec


def build_workflow(
    *,
    settings: Settings,
    prompt: str,
    steps: int,
    seed: int,
    loras: list[LoraSpec],
    reference_filenames: list[str] | None = None,
) -> dict:
    """Build a ComfyUI API-format workflow for Flux Klein text-to-image."""
    workflow: dict = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": settings.diffusion_model,
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": settings.text_encoder,
                "type": "flux2",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": settings.vae},
        },
        "4": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": settings.image_width,
                "height": settings.image_height,
                "batch_size": 1,
            },
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["2", 0],
            },
        },
        "6": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["5", 0]},
        },
        "7": {
            "class_type": "FluxGuidance",
            "inputs": {
                "conditioning": ["5", 0],
                "guidance": settings.flux_guidance,
            },
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": settings.cfg,
                "sampler_name": settings.sampler_name,
                "scheduler": settings.scheduler,
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["7", 0],
                "negative": ["6", 0],
                "latent_image": ["4", 0],
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["8", 0],
                "vae": ["3", 0],
            },
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "api_temp",
                "images": ["9", 0],
            },
        },
    }

    model_source = "1"
    clip_source = "2"
    clip_output_index = 0

    for index, lora in enumerate(loras):
        node_id = str(11 + index)
        workflow[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": [model_source, 0],
                "clip": [clip_source, clip_output_index],
                "lora_name": lora.filename,
                "strength_model": lora.weight,
                "strength_clip": lora.weight,
            },
        }
        model_source = node_id
        clip_source = node_id
        clip_output_index = 1

    if loras:
        workflow["5"]["inputs"]["clip"] = [clip_source, clip_output_index]
        workflow["8"]["inputs"]["model"] = [model_source, 0]

    positive_source = "7"
    next_node_id = 11 + len(loras)

    if reference_filenames:
        for filename in reference_filenames:
            load_id = str(next_node_id)
            scale_id = str(next_node_id + 1)
            encode_id = str(next_node_id + 2)
            ref_latent_id = str(next_node_id + 3)

            workflow[load_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": filename},
            }
            workflow[scale_id] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image": [load_id, 0],
                    "upscale_method": "lanczos",
                    "width": settings.image_width,
                    "height": settings.image_height,
                    "crop": "center",
                },
            }
            workflow[encode_id] = {
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": [scale_id, 0],
                    "vae": ["3", 0],
                },
            }
            workflow[ref_latent_id] = {
                "class_type": "ReferenceLatent",
                "inputs": {
                    "conditioning": [positive_source, 0],
                    "latent": [encode_id, 0],
                },
            }
            positive_source = ref_latent_id
            next_node_id += 4

        workflow["8"]["inputs"]["positive"] = [positive_source, 0]

    return workflow
