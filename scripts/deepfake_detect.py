#!/usr/bin/env python3
"""
Deepfake audio detection pipeline using a pre-trained HuggingFace model.
Produces: global score, per-frame timeline, mel spectrogram with overlay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Primary model — change to backup if this doesn't generalize well on Romanian audio
MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"
# Backup options:
#   "mo-thecreator/Deepfake-audio-detection"
#   "Heem2/audio-deepfake-detection"

SAMPLE_RATE = 16000       # Hz — wav2vec2 models expect 16kHz
WINDOW_S = 2.0            # seconds per analysis window
HOP_S = 0.5               # seconds between windows

_model = None
_extractor = None


def _load_model() -> tuple:
    global _model, _extractor
    if _model is None:
        print(f"Loading model: {MODEL_ID}  (first call only)")
        _extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        _model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
        _model.eval()
    return _model, _extractor


def _fake_probability(window: np.ndarray, model, extractor) -> float:
    """Run model on a single audio window, return P(AI-generated)."""
    inputs = extractor(
        window,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().numpy()

    # Find the index that corresponds to "fake/spoof/ai" label
    id2label: dict = model.config.id2label
    fake_idx = next(
        (
            int(i)
            for i, lbl in id2label.items()
            if any(k in lbl.lower() for k in ("fake", "spoof", "ai", "synth", "generated"))
        ),
        1,  # fallback: assume label index 1 is "fake"
    )
    return float(probs[fake_idx])


def detect_deepfake(
    audio_path: str | Path,
) -> Tuple[float, np.ndarray, np.ndarray, plt.Figure, plt.Figure]:
    """
    Analyze audio for AI-generation probability.

    Returns:
        global_score   : float 0-1, overall probability of being AI-generated
        timestamps     : np.ndarray of time points (seconds)
        frame_scores   : np.ndarray of per-frame probabilities
        fig_timeline   : matplotlib Figure — probability vs time
        fig_spectrogram: matplotlib Figure — mel spectrogram + overlay
    """
    model, extractor = _load_model()

    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

    win_samples = int(WINDOW_S * SAMPLE_RATE)
    hop_samples = int(HOP_S * SAMPLE_RATE)

    timestamps: list[float] = []
    frame_scores: list[float] = []

    for start in range(0, max(1, len(audio) - win_samples + 1), hop_samples):
        window = audio[start : start + win_samples]
        if len(window) < win_samples:
            # Pad last window
            window = np.pad(window, (0, win_samples - len(window)))
        score = _fake_probability(window, model, extractor)
        timestamps.append((start + win_samples // 2) / SAMPLE_RATE)
        frame_scores.append(score)

    # Edge case: very short audio (shorter than one window)
    if not frame_scores:
        padded = np.pad(audio, (0, win_samples - len(audio)))
        score = _fake_probability(padded, model, extractor)
        timestamps = [len(audio) / (2 * SAMPLE_RATE)]
        frame_scores = [score]

    timestamps_arr = np.array(timestamps)
    frame_scores_arr = np.array(frame_scores)

    # Global score: Hann-weighted average (edges matter less)
    weights = np.hanning(len(frame_scores_arr)) + 0.1
    global_score = float(np.average(frame_scores_arr, weights=weights))

    fig_timeline = _plot_timeline(timestamps_arr, frame_scores_arr, global_score)
    fig_spectrogram = _plot_spectrogram_overlay(audio, timestamps_arr, frame_scores_arr)

    return global_score, timestamps_arr, frame_scores_arr, fig_timeline, fig_spectrogram


def _plot_timeline(
    timestamps: np.ndarray, scores: np.ndarray, global_score: float
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3))

    ax.fill_between(timestamps, scores, alpha=0.25, color="#e74c3c")
    ax.plot(timestamps, scores, color="#e74c3c", linewidth=1.8, label="Frame score")
    ax.axhline(
        y=global_score,
        color="#c0392b",
        linestyle="--",
        linewidth=1.5,
        label=f"Global: {global_score:.1%}",
    )
    ax.axhline(y=0.5, color="#7f8c8d", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.text(timestamps[-1] * 0.01, 0.52, "threshold 50%", fontsize=8, color="#7f8c8d")

    ax.set_xlabel("Timp (secunde)")
    ax.set_ylabel("P(AI-generat)")
    ax.set_ylim(0, 1)
    ax.set_xlim(timestamps[0], timestamps[-1])
    verdict = "🔴 Probabil AI-generat" if global_score > 0.5 else "🟢 Probabil real"
    ax.set_title(f"Timeline Detecție Deepfake — {verdict} ({global_score:.1%})")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig


def _plot_spectrogram_overlay(
    audio: np.ndarray,
    timestamps: np.ndarray,
    scores: np.ndarray,
) -> plt.Figure:
    fig, (ax_spec, ax_prob) = plt.subplots(
        2, 1,
        figsize=(12, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # Mel spectrogram
    S = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(
        S_dB, sr=SAMPLE_RATE, x_axis="time", y_axis="mel",
        fmax=8000, ax=ax_spec, cmap="magma",
    )
    fig.colorbar(img, ax=ax_spec, format="%+2.0f dB")

    # Overlay colored spans: green=real, red=fake
    for t, score in zip(timestamps, scores):
        t_start = t - HOP_S / 2
        t_end = t + HOP_S / 2
        # RdYlGn reversed: score=1 (fake) → red, score=0 (real) → green
        color = plt.cm.RdYlGn_r(score)
        ax_spec.axvspan(t_start, t_end, alpha=0.30, color=color, linewidth=0)

    ax_spec.set_title("Mel Spectrogram cu Overlay Deepfake  (roșu = suspect)")
    ax_spec.set_ylabel("Frecvență (Hz)")

    # Probability timeline (bottom panel)
    ax_prob.plot(timestamps, scores, color="#e74c3c", linewidth=1.5)
    ax_prob.fill_between(timestamps, scores, alpha=0.25, color="#e74c3c")
    ax_prob.axhline(y=0.5, color="#7f8c8d", linestyle=":", linewidth=1.0, alpha=0.8)
    ax_prob.set_ylim(0, 1)
    ax_prob.set_ylabel("P(fake)")
    ax_prob.set_xlabel("Timp (secunde)")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect AI-generated audio")
    parser.add_argument("audio_file", help="Path to .wav file")
    parser.add_argument("--output-dir", default=".", help="Where to save plots")
    args = parser.parse_args()

    score, ts, fs, fig_tl, fig_sp = detect_deepfake(args.audio_file)
    verdict = "AI-generat" if score > 0.5 else "Real"
    print(f"\nScor global: {score:.1%}  ({verdict})")

    out = Path(args.output_dir)
    fig_tl.savefig(out / "timeline.png", dpi=150, bbox_inches="tight")
    fig_sp.savefig(out / "spectrogram_overlay.png", dpi=150, bbox_inches="tight")
    print(f"Plots saved → {out}")
