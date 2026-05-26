from __future__ import annotations

import base64
import binascii
import re
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import requests
from PIL import Image, UnidentifiedImageError

from app.config import Settings

DATA_URI_PATTERN = re.compile(r"^data:image/[\w+.-]+;base64,", re.IGNORECASE)
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}


class ReferenceImageError(ValueError):
    pass


def _decode_data_uri(value: str) -> bytes:
    payload = DATA_URI_PATTERN.sub("", value.strip())
    try:
        return base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise ReferenceImageError("Invalid base64 reference image") from exc


def _decode_base64(value: str) -> bytes:
    try:
        return base64.b64decode(value.strip(), validate=True)
    except binascii.Error as exc:
        raise ReferenceImageError("Invalid base64 reference image") from exc


def _download_url(value: str, settings: Settings) -> bytes:
    try:
        response = requests.get(
            value,
            timeout=settings.reference_download_timeout,
            stream=True,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ReferenceImageError(f"Failed to download reference image: {exc}") from exc

    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) > settings.max_reference_image_bytes:
                raise ReferenceImageError(
                    f"Reference image exceeds {settings.max_reference_image_bytes} bytes"
                )
        except ValueError:
            pass

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=65536):
        if not chunk:
            continue
        total += len(chunk)
        if total > settings.max_reference_image_bytes:
            raise ReferenceImageError(
                f"Reference image exceeds {settings.max_reference_image_bytes} bytes"
            )
        chunks.append(chunk)

    if not chunks:
        raise ReferenceImageError("Downloaded reference image is empty")

    return b"".join(chunks)


def _load_image_bytes(value: str, settings: Settings) -> bytes:
    stripped = value.strip()
    if not stripped:
        raise ReferenceImageError("Reference image value cannot be empty")

    if stripped.lower().startswith(("http://", "https://")):
        return _download_url(stripped, settings)

    if DATA_URI_PATTERN.match(stripped):
        return _decode_data_uri(stripped)

    return _decode_base64(stripped)


def _validate_image(image_bytes: bytes, settings: Settings) -> Image.Image:
    if len(image_bytes) > settings.max_reference_image_bytes:
        raise ReferenceImageError(
            f"Reference image exceeds {settings.max_reference_image_bytes} bytes"
        )

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except UnidentifiedImageError as exc:
        raise ReferenceImageError("Unsupported or invalid reference image format") from exc

    if image.format not in ALLOWED_IMAGE_FORMATS:
        raise ReferenceImageError(
            f"Unsupported reference image format: {image.format or 'unknown'}"
        )

    width, height = image.size
    max_side = max(width, height)
    if max_side > settings.max_reference_image_max_side:
        raise ReferenceImageError(
            f"Reference image exceeds max side of {settings.max_reference_image_max_side}px"
        )

    return image.convert("RGB")


def save_reference_images(settings: Settings, reference_images: list[str]) -> list[str]:
    if len(reference_images) > settings.max_reference_images:
        raise ReferenceImageError(
            f"At most {settings.max_reference_images} reference images are allowed"
        )

    input_dir = settings.comfy_input_dir
    input_dir.mkdir(parents=True, exist_ok=True)

    filenames: list[str] = []
    for value in reference_images:
        image_bytes = _load_image_bytes(value, settings)
        image = _validate_image(image_bytes, settings)

        filename = f"ref_{uuid4().hex}.png"
        output_path = input_dir / filename
        image.save(output_path, format="PNG")
        filenames.append(filename)

    return filenames


def cleanup_reference_images(settings: Settings, filenames: list[str]) -> None:
    input_dir = settings.comfy_input_dir.resolve()
    for filename in filenames:
        image_path = (input_dir / Path(filename).name).resolve()
        if input_dir not in image_path.parents:
            continue
        if image_path.is_file():
            image_path.unlink()
