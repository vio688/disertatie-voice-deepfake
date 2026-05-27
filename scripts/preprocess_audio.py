#!/usr/bin/env python3
"""
Audio preprocessing for RVC training data.
Automatically removes silences, normalizes volume, and splits into 5-10s clips.
Input:  data/processed/{voice_id}/rvc_ready/*.wav  (output of data_collection.py)
Output: data/processed/{voice_id}/clips/*.wav       (ready for RVC training)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence

TARGET_SAMPLE_RATE = 40000   # Hz — match data_collection.py (40k for RVC v2)
TARGET_DBFS = -20.0          # dBFS target for normalization
MIN_CLIP_S = 4.0             # seconds — clips shorter than this are discarded
MAX_CLIP_S = 10.0            # seconds — clips longer than this are force-split
MIN_SILENCE_MS = 400         # ms — pause shorter than this is NOT a split point
SILENCE_THRESH_OFFSET = -16  # dB below mean — what counts as "silence"


def normalize(audio: AudioSegment) -> AudioSegment:
    gain = TARGET_DBFS - audio.dBFS
    return audio.apply_gain(gain)


def split_and_save(
    audio: AudioSegment,
    output_dir: Path,
    voice_id: str,
    file_index: int,
) -> int:
    """Split audio on silence, save clips that are MIN_CLIP_S–MAX_CLIP_S long."""
    chunks = split_on_silence(
        audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=audio.dBFS + SILENCE_THRESH_OFFSET,
        keep_silence=150,
        seek_step=10,
    )

    saved = 0
    buffer = AudioSegment.empty()

    for chunk in chunks:
        buffer += chunk
        dur_s = len(buffer) / 1000.0

        if dur_s >= MIN_CLIP_S:
            if dur_s <= MAX_CLIP_S:
                _save_clip(buffer, output_dir, voice_id, file_index, saved)
                saved += 1
                buffer = AudioSegment.empty()
            else:
                # Force-cut at MAX_CLIP_S, keep the rest in buffer
                cut_ms = int(MAX_CLIP_S * 1000)
                _save_clip(buffer[:cut_ms], output_dir, voice_id, file_index, saved)
                saved += 1
                buffer = buffer[cut_ms:]

    # Flush remaining buffer if long enough
    if len(buffer) / 1000.0 >= MIN_CLIP_S:
        _save_clip(buffer, output_dir, voice_id, file_index, saved)
        saved += 1

    return saved


def _save_clip(
    clip: AudioSegment,
    output_dir: Path,
    voice_id: str,
    file_index: int,
    clip_index: int,
) -> None:
    filename = output_dir / f"{voice_id}_{file_index:04d}_{clip_index:04d}.wav"
    clip.export(
        str(filename),
        format="wav",
        parameters=["-ar", str(TARGET_SAMPLE_RATE), "-ac", "1"],
    )


def process_voice(voice_id: str, base_dir: Path) -> None:
    input_dir = base_dir / voice_id / "rvc_ready"
    output_dir = base_dir / voice_id / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        print(f"  No .wav files in {input_dir}")
        return

    print(f"  Source files: {len(wav_files)}")
    total_clips = 0

    for i, wav_file in enumerate(wav_files):
        print(f"  [{i+1}/{len(wav_files)}] {wav_file.name}")
        audio = AudioSegment.from_wav(str(wav_file))
        audio = normalize(audio)
        clips = split_and_save(audio, output_dir, voice_id, i)
        print(f"    → {clips} clips")
        total_clips += clips

    # Summary
    all_clips = list(output_dir.glob("*.wav"))
    total_duration_min = sum(
        librosa.get_duration(path=str(f)) for f in all_clips
    ) / 60.0

    print(f"\n  Total clips : {total_clips}")
    print(f"  Total audio : {total_duration_min:.1f} minutes")
    print(f"  Output dir  : {output_dir}")

    if total_duration_min < 10:
        print("  ⚠ WARNING: Less than 10 min of audio. Fine-tuning quality may suffer.")
        print("    Recommended: 15-30 min clean speech per voice.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split audio into training clips for RVC"
    )
    parser.add_argument(
        "--voice",
        default="all",
        help="Voice ID to process, or 'all' (default: all)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/processed"),
        help="Base directory containing voice subfolders",
    )
    args = parser.parse_args()

    if args.voice == "all":
        voice_dirs = [d.name for d in args.base_dir.iterdir() if d.is_dir()]
    else:
        voice_dirs = [args.voice]

    for voice_id in voice_dirs:
        print(f"\n{'='*60}")
        print(f"Preprocessing: {voice_id}")
        print(f"{'='*60}")
        process_voice(voice_id, args.base_dir)

    print("\nDone. Data is ready for RVC fine-tuning (notebooks/02_rvc_finetune.ipynb).")


if __name__ == "__main__":
    main()
