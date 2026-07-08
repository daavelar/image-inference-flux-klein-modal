from __future__ import annotations

import subprocess

DEFAULT_GPU_FALLBACK_CHAIN: list[str] = [
    "H200",
    "H100",
    "RTX-PRO-6000",
    "A100-80GB",
    "A100-40GB",
    "L40S",
]

GPU_COST_PER_SECOND: dict[str, float] = {
    "H200": 0.001261,
    "H100": 0.001097,
    "RTX-PRO-6000": 0.000842,
    "A100-80GB": 0.000694,
    "A100-40GB": 0.000583,
    "L40S": 0.000542,
}

_DEFAULT_COST = GPU_COST_PER_SECOND["H200"]


def _query_gpu_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""


def resolve_cost_per_second(gpu_name: str) -> float:
    normalized = gpu_name.upper()
    if not normalized:
        return _DEFAULT_COST
    if "H200" in normalized:
        return GPU_COST_PER_SECOND["H200"]
    if "H100" in normalized:
        return GPU_COST_PER_SECOND["H100"]
    if "RTX PRO 6000" in normalized or "RTX 6000" in normalized:
        return GPU_COST_PER_SECOND["RTX-PRO-6000"]
    if "A100" in normalized and "80" in normalized:
        return GPU_COST_PER_SECOND["A100-80GB"]
    if "A100" in normalized:
        return GPU_COST_PER_SECOND["A100-40GB"]
    if "L40S" in normalized:
        return GPU_COST_PER_SECOND["L40S"]
    return _DEFAULT_COST


def get_gpu_name() -> str:
    return _query_gpu_name()


def detect_gpu_cost_per_second() -> float:
    return resolve_cost_per_second(_query_gpu_name())
