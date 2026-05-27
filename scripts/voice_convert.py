#!/usr/bin/env python3
"""
Voice conversion wrapper for RVC v2.
Requires RVC repo cloned at RVC_DIR and fine-tuned .pth checkpoints.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# CONFIGURATION — adjust paths for your environment
# ---------------------------------------------------------------------------
RVC_DIR = Path(os.environ.get("RVC_DIR", "/content/RVC"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/content/drive/MyDrive/disertatie/models"))

# Fill in display names and transpose after fine-tuning
VOICE_CONFIG: dict[str, dict] = {
    "voice_1": {
        "name": "Călin Georgescu",
        "model_file": "voice_1.pth",
        "index_file": "voice_1.index",
        "transpose": 0,
    },
    "voice_2": {
        "name": "Nicușor Dan",
        "model_file": "voice_2.pth",
        "index_file": "voice_2.index",
        "transpose": 0,
    },
    "voice_3": {
        "name": "Diana Șoșoacă",
        "model_file": "voice_3.pth",
        "index_file": "voice_3.index",
        "transpose": 0,
    },
}
# ---------------------------------------------------------------------------


def get_available_voices() -> dict[str, str]:
    """Return {voice_id: display_name} for voices with existing .pth files."""
    return {
        vid: cfg["name"]
        for vid, cfg in VOICE_CONFIG.items()
        if (MODELS_DIR / cfg["model_file"]).exists()
    }


def _ensure_rvc_in_path() -> None:
    rvc_path = str(RVC_DIR)
    if rvc_path not in sys.path:
        sys.path.insert(0, rvc_path)


def voice_convert(
    audio_path: str | Path,
    voice_id: str,
    output_path: Optional[str | Path] = None,
    f0_method: str = "rmvpe",
    transpose: Optional[int] = None,
    index_rate: float = 0.75,
    filter_radius: int = 3,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33,
) -> str:
    """
    Convert vocal identity in audio_path to target voice.

    Args:
        audio_path   : Input .wav file
        voice_id     : Key in VOICE_CONFIG (e.g. "voice_1")
        output_path  : Output .wav path; auto-generated if None
        f0_method    : Pitch extractor — "rmvpe" (best) or "crepe"
        transpose    : Pitch shift in semitones; None uses voice default
        index_rate   : FAISS retrieval weight 0.0–1.0 (default 0.75)
        filter_radius: Median filter for pitch smoothing (1–7)
        rms_mix_rate : Output energy mix (0=pure model, 1=match input)
        protect      : Consonant protection 0.0–0.5 (default 0.33)

    Returns:
        Path to converted .wav file
    """
    if voice_id not in VOICE_CONFIG:
        raise ValueError(f"Unknown voice_id '{voice_id}'. Available: {list(VOICE_CONFIG)}")

    cfg = VOICE_CONFIG[voice_id]
    model_path = MODELS_DIR / cfg["model_file"]
    index_path = MODELS_DIR / cfg["index_file"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run notebooks/02_rvc_finetune.ipynb first, then copy .pth to Drive."
        )

    if transpose is None:
        transpose = cfg.get("transpose", 0)

    _ensure_rvc_in_path()

    # Import RVC inference API (available after cloning the repo)
    from infer.modules.vc.modules import VC  # type: ignore
    from configs.config import Config         # type: ignore

    config = Config()
    vc = VC(config)
    vc.get_vc(str(model_path))

    if output_path is None:
        audio_path = Path(audio_path)
        output_path = audio_path.with_name(f"{audio_path.stem}_{voice_id}.wav")

    tgt_sr, output_audio = vc.vc_single(
        sid=0,
        input_audio_path=str(audio_path),
        f0_up_key=transpose,
        f0_file=None,
        f0_method=f0_method,
        file_index=str(index_path) if index_path.exists() else "",
        index_rate=index_rate,
        filter_radius=filter_radius,
        resample_sr=0,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
    )

    sf.write(str(output_path), output_audio, tgt_sr)
    return str(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert voice using RVC v2")
    parser.add_argument("audio_file", help="Input .wav file")
    parser.add_argument("voice_id", choices=list(VOICE_CONFIG), help="Target voice")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--transpose", type=int, default=None)
    parser.add_argument("--f0-method", default="rmvpe", choices=["rmvpe", "crepe"])
    parser.add_argument("--index-rate", type=float, default=0.75)
    args = parser.parse_args()

    out = voice_convert(
        args.audio_file,
        args.voice_id,
        output_path=args.output,
        transpose=args.transpose,
        f0_method=args.f0_method,
        index_rate=args.index_rate,
    )
    print(f"Output: {out}")
