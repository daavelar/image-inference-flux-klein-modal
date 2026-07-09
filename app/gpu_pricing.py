from __future__ import annotations

from app.modal_billing import (
    GPU_COST_PER_SECOND,
    detect_gpu_cost_per_second,
    get_gpu_name,
    resolve_gpu_cost_per_second,
)

DEFAULT_GPU_FALLBACK_CHAIN: list[str] = [
    "H200",
    "H100",
    "RTX-PRO-6000",
    "A100-80GB",
    "A100-40GB",
    "L40S",
]

_DEFAULT_COST = GPU_COST_PER_SECOND["H200"]


def resolve_cost_per_second(gpu_name: str) -> float:
    return resolve_gpu_cost_per_second(gpu_name)[1]


__all__ = [
    "DEFAULT_GPU_FALLBACK_CHAIN",
    "GPU_COST_PER_SECOND",
    "detect_gpu_cost_per_second",
    "get_gpu_name",
    "resolve_cost_per_second",
]
