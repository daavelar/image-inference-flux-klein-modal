from __future__ import annotations

import logging
import os
import sys

_NOISY_LOGGERS = (
    "comfy",
    "execution",
    "folder_paths",
    "nodes",
    "comfy_extras",
    "comfy_api",
)


def configure_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    if logging.root.handlers:
        logging.root.setLevel(level)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def configure_aimdo_logging() -> None:
    try:
        import comfy_aimdo.control as aimdo
    except ImportError:
        return

    if aimdo.lib is None:
        return

    level = os.environ.get("AIMDO_LOG_LEVEL", "none").strip().lower()
    setters = {
        "none": aimdo.set_log_none,
        "critical": aimdo.set_log_critical,
        "error": aimdo.set_log_error,
        "warning": aimdo.set_log_warning,
        "info": aimdo.set_log_info,
        "debug": aimdo.set_log_debug,
        "verbose": aimdo.set_log_verbose,
        "vverbose": aimdo.set_log_vverbose,
    }
    setter = setters.get(level, aimdo.set_log_none)
    setter()
