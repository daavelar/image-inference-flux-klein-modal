#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-$HOME/Projetos/ComfyUI}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/vendor/comfyui"

echo "Syncing ComfyUI runtime from $SRC to $DEST"

rsync -a --delete \
  --exclude='.venv' --exclude='.git' --exclude='__pycache__' \
  --exclude='custom_nodes' --exclude='models' --exclude='output' \
  --exclude='input' --exclude='user' \
  "$SRC/comfy/" "$DEST/comfy/"

for dir in comfy_execution comfy_api comfy_extras comfy_config utils app middleware; do
  rsync -a --delete "$SRC/$dir/" "$DEST/$dir/"
done

cp "$SRC"/{nodes.py,execution.py,folder_paths.py,latent_preview.py,node_helpers.py,cuda_malloc.py,hook_breaker_ac10a0.py,comfyui_version.py,protocol.py} "$DEST/"

echo "Done."
