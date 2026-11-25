"""Audio Preprocessing Module: Resampling, Mono Conversion, Normalization, Noise Analysis."""

from typing import Tuple, Dict, Any
import numpy as np
import scipy.signal


def to_mono(signal: np.ndarray) -> np.ndarray:
    """Averages channels if multi-channel array is provided."""
    if signal.ndim == 1:
        return signal
    return np.mean(signal, axis=1)


def normalize_peak(signal: np.ndarray, target_level: float = 0.95) -> np.ndarray:
    """
    Peak normalization to prevent clipping and ensure consistent dynamic range.
    Avoids dividing by zero on silence.
    """
    max_amp = np.max(np.abs(signal))
    if max_amp > 1e-6:
        return signal * (target_level / max_amp)
    return signal


def pre_emphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    Pre-emphasis filter: y[n] = x[n] - a * x[n-1]
    Boosts higher frequencies to compensate for the natural spectral roll-off (~-6 dB/octave) of human speech.
    Standard course concept in Lawrence Rabiner's speech processing syllabus.
    """
    if len(signal) == 0:
        return signal
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def analyze_noise(signal: np.ndarray, sr: int = 16000, initial_duration: float = 0.5) -> Dict[str, Any]:
    """
    Estimates background noise baseline from the initial presumed-silent frames.
    Computes RMS noise floor, SNR estimation, and dynamic range.
    """
    sample_count = int(min(len(signal), sr * initial_duration))
    noise_segment = signal[:sample_count] if sample_count > 0 else signal

    noise_rms = float(np.sqrt(np.mean(noise_segment**2))) if len(noise_segment) > 0 else 1e-6
    signal_rms = float(np.sqrt(np.mean(signal**2))) if len(signal) > 0 else 1e-6

    snr_db = 20 * np.log10(max(signal_rms, 1e-6) / max(noise_rms, 1e-6))

    return {
        "noise_floor_rms": round(noise_rms, 5),
        "signal_rms": round(signal_rms, 5),
        "estimated_snr_db": round(float(snr_db), 2),
        "peak_amplitude": round(float(np.max(np.abs(signal))), 4) if len(signal) > 0 else 0.0,
    }


def preprocess_pipeline(signal: np.ndarray, sr: int = 16000) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Standard lecture processing pipeline:
    1. Mono conversion
    2. Peak normalization
    3. Noise characteristics extraction
    """
    mono_sig = to_mono(signal)
    norm_sig = normalize_peak(mono_sig)
    noise_stats = analyze_noise(norm_sig, sr=sr)
    return norm_sig, noise_stats
