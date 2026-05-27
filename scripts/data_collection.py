#!/usr/bin/env python3
"""
Download audio from YouTube and prepare it for RVC fine-tuning.
Pipeline: yt-dlp download → Demucs vocal separation → ffmpeg 16kHz mono WAV.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION — fill in before running
# ---------------------------------------------------------------------------
VOICES: dict[str, dict] = {
    "voice_1": {
        "name": "Călin Georgescu",
        "youtube_urls": [
            # Adaugă link-uri YouTube cu discursuri/videoclipuri solo
            # Preferă videoclipuri unde vorbește singur (nu interviuri cu moderator)
            # Ex: "https://www.youtube.com/watch?v=XXXXXXXXX",
        ],
    },
    "voice_2": {
        "name": "Nicușor Dan",
        "youtube_urls": [
            # Conferințe de presă, declarații solo
        ],
    },
    "voice_3": {
        "name": "Diana Șoșoacă",
        "youtube_urls": [
            # Discursuri la parlament — cele mai curate surse audio
        ],
    },
}

DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")

# Demucs model — htdemucs_ft is best quality but slower; htdemucs is faster
DEMUCS_MODEL = "htdemucs_ft"
# ---------------------------------------------------------------------------


def download_audio(voice_id: str, urls: list[str], output_dir: Path) -> None:
    voice_dir = output_dir / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)

    for url in urls:
        print(f"  Downloading: {url}")
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--output", str(voice_dir / "%(title)s.%(ext)s"),
            "--no-playlist",
            url,
        ]
        subprocess.run(cmd, check=True)


def separate_vocals(voice_id: str, raw_dir: Path, processed_dir: Path) -> None:
    input_dir = raw_dir / voice_id
    output_parent = processed_dir / voice_id / "demucs_out"

    wav_files = list(input_dir.glob("*.wav"))
    if not wav_files:
        print(f"  No .wav files in {input_dir}, skipping.")
        return

    for wav_file in wav_files:
        print(f"  Separating vocals: {wav_file.name}")
        cmd = [
            "python", "-m", "demucs",
            f"--two-stems=vocals",
            "--model", DEMUCS_MODEL,
            "--out", str(output_parent),
            str(wav_file),
        ]
        subprocess.run(cmd, check=True)


def convert_to_rvc_format(voice_id: str, processed_dir: Path) -> None:
    """Convert separated vocals to 16kHz mono PCM WAV (RVC requirement)."""
    demucs_out = processed_dir / voice_id / "demucs_out"
    rvc_ready = processed_dir / voice_id / "rvc_ready"
    rvc_ready.mkdir(parents=True, exist_ok=True)

    # Demucs output: {demucs_out}/{model_name}/{original_stem}/vocals.wav
    vocals_found = list(demucs_out.rglob("vocals.wav"))
    if not vocals_found:
        print(f"  No vocals.wav found under {demucs_out}")
        return

    for vocals_file in vocals_found:
        stem_name = vocals_file.parent.name
        output_file = rvc_ready / f"{stem_name}.wav"
        print(f"  Converting: {vocals_file} → {output_file.name}")
        cmd = [
            "ffmpeg",
            "-i", str(vocals_file),
            "-ar", "40000",   # RVC v2 prefers 40kHz; change to 16000 if needed
            "-ac", "1",
            "-acodec", "pcm_s16le",
            str(output_file),
            "-y",
        ]
        subprocess.run(cmd, check=True)

    count = len(list(rvc_ready.glob("*.wav")))
    print(f"  RVC-ready files: {count} → {rvc_ready}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess voice data for RVC fine-tuning"
    )
    parser.add_argument(
        "--voice",
        choices=[*VOICES.keys(), "all"],
        default="all",
        help="Which voice to process (default: all)",
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip yt-dlp download step"
    )
    parser.add_argument(
        "--skip-demucs", action="store_true", help="Skip Demucs vocal separation"
    )
    parser.add_argument(
        "--skip-convert", action="store_true", help="Skip ffmpeg conversion step"
    )
    args = parser.parse_args()

    voices_to_process = list(VOICES.keys()) if args.voice == "all" else [args.voice]

    for voice_id in voices_to_process:
        config = VOICES[voice_id]
        print(f"\n{'='*60}")
        print(f"Processing: {config['name']} ({voice_id})")
        print(f"{'='*60}")

        if not args.skip_download:
            if not config["youtube_urls"]:
                print("  WARNING: No YouTube URLs configured. Add them to VOICES dict.")
                continue
            download_audio(voice_id, config["youtube_urls"], DATA_RAW_DIR)

        if not args.skip_demucs:
            separate_vocals(voice_id, DATA_RAW_DIR, DATA_PROCESSED_DIR)

        if not args.skip_convert:
            convert_to_rvc_format(voice_id, DATA_PROCESSED_DIR)

    print("\nDone. Next step: run preprocess_audio.py to split into 5-10s clips.")


if __name__ == "__main__":
    main()
