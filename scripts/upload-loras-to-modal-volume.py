#!/usr/bin/env python3
"""
Upload local LoRA weights into a Modal Volume for generate-modal.py.

Expected layout inside the volume:
    /models/loras/<filename>.safetensors

Environment (optional):
    MODAL_VOLUME_NAME   — default fluxkleininf
    MODAL_MODELS_REMOTE — models root inside the volume, default /models

CLI:
    pip install modal
    modal setup  # once
    python scripts/upload-loras-to-modal-volume.py
    python scripts/upload-loras-to-modal-volume.py loras/my_lora.safetensors
    python scripts/upload-loras-to-modal-volume.py --force

Or use Modal CLI:
    modal volume put fluxkleininf ./loras/my_lora.safetensors /models/loras/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LORAS_DIR = REPO_ROOT / "loras"
LORA_SUFFIXES = (".safetensors",)


def _collect_lora_files(local_dir: Path, paths: list[Path]) -> list[Path]:
    if paths:
        files: list[Path] = []
        for raw in paths:
            candidate = raw.expanduser()
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve()
            else:
                candidate = candidate.resolve()

            if candidate.is_dir():
                files.extend(
                    sorted(
                        path
                        for path in candidate.iterdir()
                        if path.is_file() and path.suffix.lower() in LORA_SUFFIXES
                    )
                )
                continue

            if not candidate.is_file():
                raise FileNotFoundError(f"LoRA file not found: {candidate}")
            if candidate.suffix.lower() not in LORA_SUFFIXES:
                raise ValueError(f"Unsupported LoRA extension (expected .safetensors): {candidate.name}")
            files.append(candidate)
        return files

    if not local_dir.is_dir():
        raise FileNotFoundError(f"LoRA directory not found: {local_dir}")

    files = sorted(
        path
        for path in local_dir.iterdir()
        if path.is_file() and path.suffix.lower() in LORA_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {local_dir}")
    return files


def main(argv: list[str] | None = None) -> int:
    default_vol = os.environ.get("MODAL_VOLUME_NAME", "fluxkleininf")
    default_remote = os.environ.get("MODAL_MODELS_REMOTE", "/models").strip() or "/"

    p = argparse.ArgumentParser(description="Upload LoRA files to a Modal Volume.")
    p.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="LoRA file(s) or directory to upload (default: all .safetensors in ./loras)",
    )
    p.add_argument("--volume", "-v", default=default_vol, help="Modal volume name")
    p.add_argument(
        "--local-dir",
        "-l",
        type=Path,
        default=DEFAULT_LORAS_DIR,
        help="Local loras directory when no files are passed",
    )
    p.add_argument(
        "--remote",
        "-r",
        default=default_remote,
        help="Models root inside the volume (default /models → uploads to /models/loras/)",
    )
    p.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing files in the volume",
    )
    args = p.parse_args(argv)

    local_dir = args.local_dir.expanduser().resolve()
    models_root = args.remote.strip() or "/"
    models_root = "/" if models_root in ("", "/") else "/" + models_root.strip("/")
    remote_loras_dir = f"{models_root}/loras"

    try:
        lora_files = _collect_lora_files(local_dir, args.files)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        import modal
    except ImportError:
        print("error: install the Modal client: pip install modal", file=sys.stderr)
        return 1

    vol = modal.Volume.from_name(args.volume, create_if_missing=True)
    print(
        f"Uploading {len(lora_files)} LoRA file(s) to volume {args.volume!r} at {remote_loras_dir!r}:",
        flush=True,
    )
    for path in lora_files:
        print(f"  {path.name}", flush=True)

    with vol.batch_upload(force=args.force) as batch:
        for path in lora_files:
            batch.put_file(str(path), f"{remote_loras_dir}/{path.name}")

    print(f"Done. Example: modal volume ls {args.volume} {remote_loras_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
