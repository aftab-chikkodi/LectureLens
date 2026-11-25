"""Signal Processing and Feature Visualization Engine.
Produces genuine matplotlib figures from uploaded audio:
1. Raw & Preprocessed Waveforms
2. Short-Time Fourier Transform Spectrogram
3. Mel-Scale Spectrogram
4. MFCC Feature Heatmap
5. LPC Frequency Response and Formant Envelope
6. VAD Speech Activity Timeline
"""

from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from src.core.models import VADInterval, DiarizationSegment
from src.core.spr_algorithms import compute_lpc_levinson


def plot_waveform(
    signal: np.ndarray,
    sr: int = 16000,
    title: str = "Audio Waveform (Time Domain)",
    color: str = "#2563EB",
) -> plt.Figure:
    """Plots amplitude vs time with accurate units."""
    fig, ax = plt.subplots(figsize=(10, 2.8))
    times = np.linspace(0, len(signal) / sr, num=len(signal))
    ax.plot(times, signal, color=color, linewidth=0.7, alpha=0.85)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (normalized)")
    ax.set_xlim(0, max(times[-1] if len(times) else 1.0, 1.0))
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_spectrogram(
    signal: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
) -> plt.Figure:
    """Plots STFT Linear Spectrogram (dB)."""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="linear", ax=ax, cmap="viridis"
    )
    ax.set_title("Short-Time Fourier Transform (STFT) Spectrogram", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (seconds)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def plot_mel_spectrogram(
    signal: np.ndarray,
    sr: int = 16000,
    n_mels: int = 64,
) -> plt.Figure:
    """Plots Mel-Scale Filterbank Spectrogram."""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(
        S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma"
    )
    ax.set_title(f"Mel-Scale Spectrogram ({n_mels} Mel Filterbanks)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mel Frequency (Hz)")
    ax.set_xlabel("Time (seconds)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def plot_mfcc(
    signal: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
) -> plt.Figure:
    """Plots Mel-Frequency Cepstral Coefficients (MFCC) over time."""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    img = librosa.display.specshow(mfccs, sr=sr, x_axis="time", ax=ax, cmap="coolwarm")
    ax.set_title(f"Mel-Frequency Cepstral Coefficients (MFCC 1..{n_mfcc})", fontsize=12, fontweight="bold")
    ax.set_ylabel("MFCC Index")
    ax.set_xlabel("Time (seconds)")
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    return fig


def plot_lpc_spectrum(
    signal: np.ndarray,
    sr: int = 16000,
    order: int = 14,
    center_sec: Optional[float] = None,
) -> plt.Figure:
    """
    Computes and plots LPC spectral envelope vs FFT spectrum for a chosen 30ms speech frame.
    Highlights vocal tract resonances (formants).
    """
    fig, ax = plt.subplots(figsize=(10, 3.2))
    frame_len = int(sr * 0.030)  # 30 ms frame

    if center_sec is None or (center_sec * sr) >= len(signal):
        # Pick high energy frame
        center_idx = len(signal) // 2
    else:
        center_idx = int(center_sec * sr)

    start = max(0, center_idx - frame_len // 2)
    end = min(len(signal), start + frame_len)
    frame = signal[start:end]

    if len(frame) < frame_len:
        frame = np.pad(frame, (0, frame_len - len(frame)))

    # FFT Spectrum
    windowed = frame * np.hamming(len(frame))
    fft_spec = np.abs(np.fft.rfft(windowed, n=1024))
    fft_db = 20 * np.log10(np.maximum(fft_spec, 1e-5))
    freqs = np.linspace(0, sr / 2, len(fft_db))

    # LPC Envelope
    lpc_coeffs, error = compute_lpc_levinson(frame, order=order)
    # Frequency response of all-pole filter H(z) = G / A(z)
    w, h = scipy.signal.freqz([np.sqrt(max(error, 1e-6))], lpc_coeffs, worN=len(freqs))
    lpc_db = 20 * np.log10(np.maximum(np.abs(h), 1e-5))

    ax.plot(freqs, fft_db, label="FFT Spectrum", color="#94A3B8", alpha=0.7, linewidth=1.0)
    ax.plot(freqs, lpc_db, label=f"LPC Envelope (p={order})", color="#DC2626", linewidth=2.0)
    ax.set_title("LPC Spectral Envelope & Formant Resonances (Levinson-Durbin)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(0, sr // 2)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_vad_and_diarization_timeline(
    duration: float,
    vad_intervals: List[VADInterval],
    diar_segments: List[DiarizationSegment],
) -> plt.Figure:
    """Plots aligned timeline comparing VAD speech regions and Speaker turns."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 3.0), sharex=True)

    # 1. VAD bar
    for v in vad_intervals:
        ax1.axvspan(v.start, v.end, color="#10B981", alpha=0.7)
    ax1.set_yticks([])
    ax1.set_ylabel("VAD Speech", fontweight="bold", fontsize=10)
    ax1.set_xlim(0, max(duration, 1.0))
    ax1.grid(True, linestyle=":", alpha=0.4)

    # 2. Speaker Diarization bars
    speaker_palette = {
        "Speaker 1": "#3B82F6",
        "Speaker 2": "#F59E0B",
        "Speaker 3": "#8B5CF6",
        "Speaker 4": "#EC4899",
        "Speaker 5": "#14B8A6",
    }

    for d in diar_segments:
        color = speaker_palette.get(d.speaker_id, "#6B7280")
        ax2.axvspan(d.start, d.end, color=color, alpha=0.8)
        mid = (d.start + d.end) / 2.0
        if (d.end - d.start) > (duration * 0.05):
            ax2.text(mid, 0.5, d.speaker_id, ha="center", va="center", color="white", fontweight="bold", fontsize=8)

    ax2.set_yticks([])
    ax2.set_ylabel("Speaker Diar.", fontweight="bold", fontsize=10)
    ax2.set_xlabel("Time (seconds)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle("Temporal Alignment: Voice Activity & Speaker Diarization", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
