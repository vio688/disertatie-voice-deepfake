#!/usr/bin/env python3
"""
Speaker diarization: automatically separates speakers in a podcast/interview.
Keeps only the PRIMARY speaker (the one with the most speaking time).

Uses pyannote.audio v3 — requires a FREE HuggingFace token.
Setup (one time only):
  1. Create account: https://huggingface.co/
  2. Accept license: https://hf.co/pyannote/speaker-diarization-3.1
  3. Accept license: https://hf.co/pyannote/segmentation-3.0
  4. Create token: https://hf.co/settings/tokens (read-only is fine)
  5. Set env var: HF_TOKEN=your_token_here  (or paste it below)
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
import librosa

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")   # Set env var, or paste token here

# Which speaker to keep: "largest" = most speaking time (usually the main guest)
KEEP_SPEAKER = "largest"

# Minimum segment duration to keep (short utterances are noise)
MIN_SEGMENT_S = 1.5

# Output: 40kHz mono WAV (RVC format)
OUTPUT_SAMPLE_RATE = 40000
# ---------------------------------------------------------------------------


def diarize_and_extract(
    audio_path: str | Path,
    output_path: str | Path,
    n_speakers: int = 2,
    keep: str = "largest",
) -> dict:
    """
    Diarize audio, extract segments of the primary speaker.

    Args:
        audio_path  : Input audio file
        output_path : Output WAV with only the primary speaker
        n_speakers  : Expected number of speakers (2 for interview, 1+ for panel)
        keep        : "largest" = speaker with most time, or speaker label (e.g. "SPEAKER_01")

    Returns:
        Dict with stats: n_speakers_found, kept_speaker, kept_duration_s, total_duration_s
    """
    if not HF_TOKEN:
        raise EnvironmentError(
            "HuggingFace token not set.\n"
            "Set env var HF_TOKEN=... or edit HF_TOKEN in speaker_diarize.py.\n"
            "Get token at: https://huggingface.co/settings/tokens"
        )

    from pyannote.audio import Pipeline

    print(f"Loading pyannote diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    )
    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device("cuda"))

    audio_path = Path(audio_path)
    print(f"Diarizing: {audio_path.name}")

    diarization = pipeline(str(audio_path), num_speakers=n_speakers)

    # Collect segments per speaker
    speaker_segments: dict[str, list[tuple[float, float]]] = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        duration = turn.end - turn.start
        if duration < MIN_SEGMENT_S:
            continue
        speaker_segments.setdefault(speaker, []).append((turn.start, turn.end))

    if not speaker_segments:
        raise RuntimeError("No speech segments found after diarization.")

    # Determine which speaker to keep
    if keep == "largest":
        kept_speaker = max(
            speaker_segments,
            key=lambda s: sum(e - st for st, e in speaker_segments[s]),
        )
    else:
        if keep not in speaker_segments:
            available = list(speaker_segments.keys())
            raise ValueError(f"Speaker '{keep}' not found. Available: {available}")
        kept_speaker = keep

    print(f"\nSpeakers found: {list(speaker_segments.keys())}")
    for sp, segs in speaker_segments.items():
        total = sum(e - st for st, e in segs)
        marker = " ← KEEPING" if sp == kept_speaker else ""
        print(f"  {sp}: {total:.1f}s in {len(segs)} segments{marker}")

    # Load audio and extract kept-speaker segments
    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)

    kept_segments = speaker_segments[kept_speaker]
    kept_parts = []
    for start_s, end_s in sorted(kept_segments):
        start_i = int(start_s * sr)
        end_i = int(end_s * sr)
        kept_parts.append(audio[start_i:end_i])

    if not kept_parts:
        raise RuntimeError("No audio segments found for target speaker.")

    # Concatenate with small silence padding between segments
    silence_pad = np.zeros(int(0.1 * sr), dtype=np.float32)
    combined = np.concatenate(
        [np.concatenate([part, silence_pad]) for part in kept_parts]
    )

    # Resample to RVC format
    if sr != OUTPUT_SAMPLE_RATE:
        combined = librosa.resample(combined, orig_sr=sr, target_sr=OUTPUT_SAMPLE_RATE)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), combined, OUTPUT_SAMPLE_RATE)

    kept_duration = sum(e - st for st, e in kept_segments)
    total_duration = len(audio) / sr

    return {
        "n_speakers_found": len(speaker_segments),
        "kept_speaker": kept_speaker,
        "kept_duration_s": kept_duration,
        "total_duration_s": total_duration,
        "output": str(output_path),
    }


def process_voice_folder(
    voice_id: str,
    input_dir: Path,
    output_dir: Path,
    n_speakers: int = 2,
) -> None:
    """Process all audio files for a voice ID."""
    wav_files = sorted(input_dir.glob("*.wav")) + sorted(input_dir.glob("*.mp3"))

    if not wav_files:
        print(f"  No audio files in {input_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Diarizing: {voice_id}  ({len(wav_files)} files)")
    print(f"{'='*60}")

    for audio_file in wav_files:
        out_file = output_dir / audio_file.name
        try:
            stats = diarize_and_extract(
                audio_file,
                out_file,
                n_speakers=n_speakers,
            )
            kept_pct = stats["kept_duration_s"] / stats["total_duration_s"] * 100
            print(
                f"\n  ✓ {audio_file.name} → {stats['kept_duration_s']/60:.1f} min "
                f"of primary speaker ({kept_pct:.0f}% of total)\n"
            )
        except Exception as e:
            print(f"\n  ✗ Failed: {audio_file.name}: {e}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate speakers in podcast audio — keep only primary speaker"
    )
    parser.add_argument(
        "input_dir",
        help="Folder with audio files (e.g. data/raw/voice_1/)",
    )
    parser.add_argument(
        "output_dir",
        help="Folder for diarized output (e.g. data/processed/voice_1/rvc_ready/)",
    )
    parser.add_argument("--voice-id", default="voice", help="Voice ID label")
    parser.add_argument(
        "--n-speakers", type=int, default=2,
        help="Expected number of speakers (default: 2 for interview)",
    )
    parser.add_argument(
        "--token", default="",
        help="HuggingFace token (overrides HF_TOKEN env var)",
    )
    args = parser.parse_args()

    if args.token:
        global HF_TOKEN
        HF_TOKEN = args.token

    process_voice_folder(
        voice_id=args.voice_id,
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        n_speakers=args.n_speakers,
    )

    print("\nDone. Continuă cu preprocess_audio.py pentru split în clipuri 5-10s.")


if __name__ == "__main__":
    main()
