"""HW3 guidance package (SDS / PDS).

Runs before `guidance.sd` loads so Colab can fix stale starters that hardcode
`stabilityai/stable-diffusion-2-1-base` (Hub 404) even when `main.py` was never updated.
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_HF_SD21 = "sd2-community/stable-diffusion-2-1-base"
os.environ.setdefault("SD_MODEL_ID", _DEFAULT_HF_SD21)


def _patch_sd_py_if_stale_hub_id() -> None:
    p = Path(__file__).resolve().parent / "sd.py"
    try:
        text = p.read_text()
    except OSError:
        return
    if "stabilityai/stable-diffusion-2-1-base" not in text:
        return
    if "_HF_SD21_MIRROR" in text:
        return
    new = text.replace(
        "stabilityai/stable-diffusion-2-1-base",
        "sd2-community/stable-diffusion-2-1-base",
    )
    if new == text:
        return
    p.write_text(new)
    print(
        "[WARN] guidance: patched sd.py Hub id (stabilityai -> sd2-community). "
        "Run `git pull` for the full upstream files."
    )


_patch_sd_py_if_stale_hub_id()
