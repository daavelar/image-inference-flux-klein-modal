#!/usr/bin/env python3
"""
Upload local Flux Klein model weights into a Modal Volume for generate-modal.py.

Expected layout inside the volume:
    /models/diffusion_models/
    /models/text_encoders/
    /models/vae/
    /models/loras/

Environment (optional):
    MODAL_VOLUME_NAME   — default fluxkleininf
    MODAL_MODELS_LOCAL  — default <repo>/models (built from repo model dirs)
    MODAL_MODELS_REMOTE — path inside the volume root, default /models

CLI:
    pip install modal
    modal setup  # once
    python scripts/upload-models-to-modal-volume.py
    python scripts/upload-models-to-modal-volume.py --local ./models --remote /models

Or use Modal CLI:
    modal volume put fluxkleininf ./models /models
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_SUBDIRS = ("diffusion_models", "text_encoders", "vae", "loras")


def _build_local_models_tree(source_root: Path, dest_root: Path) -> None:
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    for subdir in MODEL_SUBDIRS:
        src = source_root / subdir
        if not src.is_dir():
            raise FileNotFoundError(f"Missing required directory: {src}")
        shutil.copytree(src, dest_root / subdir)


def main(argv: list[str] | None = None) -> int:
    default_vol = os.environ.get("MODAL_VOLUME_NAME", "fluxkleininf")
    default_local = Path(os.environ.get("MODAL_MODELS_LOCAL", REPO_ROOT / "models"))
    default_remote = os.environ.get("MODAL_MODELS_REMOTE", "/models").strip() or "/"

    p = argparse.ArgumentParser(description="Upload Flux Klein model tree to a Modal Volume.")
    p.add_argument("--volume", "-v", default=default_vol, help="Modal volume name")
    p.add_argument(
        "--local",
        "-l",
        type=Path,
        default=default_local,
        help="Local directory containing model subfolders (or repo root to assemble)",
    )
    p.add_argument(
        "--remote",
        "-r",
        default=default_remote,
        help="Destination path inside the volume (e.g. /models)",
    )
    p.add_argument(
        "--from-repo",
        action="store_true",
        help="Assemble ./models from repo diffusion_models/, text_encoders/, vae/, loras/",
    )
    p.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing files in the volume",
    )
    args = p.parse_args(argv)

    local = args.local.expanduser().resolve()
    upload_root = local
    temp_dir: tempfile.TemporaryDirectory[str] | None = None

    if args.from_repo or not local.is_dir():
        temp_dir = tempfile.TemporaryDirectory(prefix="fluxklein-models-")
        upload_root = Path(temp_dir.name) / "models"
        try:
            _build_local_models_tree(REPO_ROOT, upload_root)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    for subdir in MODEL_SUBDIRS:
        if not (upload_root / subdir).is_dir():
            print(f"error: missing {upload_root / subdir}", file=sys.stderr)
            return 1

    r = args.remote.strip() or "/"
    dest = "/" if r in ("", "/") else "/" + r.strip("/")

    try:
        import modal
    except ImportError:
        print("error: install the Modal client: pip install modal", file=sys.stderr)
        return 1

    vol = modal.Volume.from_name(args.volume, create_if_missing=True)
    print(f"Uploading:\n  {upload_root}\n→ volume {args.volume!r} at {dest!r}")
    print("(large uploads can take a long time)", flush=True)

    local_str = str(upload_root) + ("" if str(upload_root).endswith("/") else "/")
    with vol.batch_upload(force=args.force) as batch:
        batch.put_directory(local_str, dest)

    print(f"Done. Example: modal volume ls {args.volume} {dest}")
    if temp_dir is not None:
        temp_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
