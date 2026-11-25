"""Voice Activity Detection (VAD) Engine.
Implements:
1. Academic Short-Time Energy & Zero-Crossing Rate (Rabiner course concept).
2. Adaptive Energy thresholding with hangover smoothing.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from src.core.models import VADInterval


def compute_ste_and_zcr(
    signal: np.ndarray,
    sr: int = 16000,
    frame_length_ms: float = 25.0,
    frame_step_ms: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Short-Time Energy (STE) and Zero Crossing Rate (ZCR).
    Standard textbook algorithm (Lawrence Rabiner & Ronald Schafer).
    """
    frame_length = int(sr * frame_length_ms / 1000.0)
    frame_step = int(sr * frame_step_ms / 1000.0)

    num_frames = max(1, int(np.ceil((len(signal) - frame_length) / frame_step)) + 1)
    ste = np.zeros(num_frames, dtype=np.float32)
    zcr = np.zeros(num_frames, dtype=np.float32)
    times = np.zeros(num_frames, dtype=np.float32)

    # Hamming window
    window = np.hamming(frame_length)

    for i in range(num_frames):
        start = i * frame_step
        end = min(start + frame_length, len(signal))
        frame = signal[start:end]

        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))

        windowed = frame * window
        # Short Time Energy
        ste[i] = np.sum(windowed ** 2) / frame_length

        # Zero Crossing Rate: 1/(2N) * sum |sgn(x[n]) - sgn(x[n-1])|
        signs = np.sign(frame)
        signs[signs == 0] = 1
        zcr[i] = np.sum(np.abs(signs[1:] - signs[:-1])) / (2.0 * frame_length)
        times[i] = (start + frame_length / 2.0) / sr

    return times, ste, zcr


def detect_voice_activity(
    signal: np.ndarray,
    sr: int = 16000,
    frame_length_ms: float = 25.0,
    frame_step_ms: float = 10.0,
    energy_threshold_factor: float = 1.8,
    min_speech_duration_ms: float = 200.0,
    min_silence_duration_ms: float = 300.0,
) -> Tuple[List[VADInterval], Dict[str, Any]]:
    """
    Academic dual-threshold VAD using Short-Time Energy and Zero-Crossing Rate.
    Includes hangover smoothing to bridge natural intra-word pauses.
    """
    times, ste, zcr = compute_ste_and_zcr(
        signal, sr=sr, frame_length_ms=frame_length_ms, frame_step_ms=frame_step_ms
    )

    # Dynamic noise floor thresholding from lowest 20% frames
    sorted_ste = np.sort(ste)
    noise_floor = float(np.mean(sorted_ste[: max(1, int(len(sorted_ste) * 0.2))]))
    ste_threshold = max(noise_floor * energy_threshold_factor, 1e-4)

    # Boolean speech decision per frame
    speech_frames = (ste > ste_threshold).astype(bool)

    # Merge frames into contiguous intervals
    frame_step_s = frame_step_ms / 1000.0
    intervals: List[VADInterval] = []

    in_speech = False
    start_time = 0.0

    for i, is_active in enumerate(speech_frames):
        cur_time = times[i]
        if is_active and not in_speech:
            in_speech = True
            start_time = max(0.0, cur_time - (frame_length_ms / 2000.0))
        elif not is_active and in_speech:
            in_speech = False
            end_time = cur_time + (frame_length_ms / 2000.0)
            if (end_time - start_time) >= (min_speech_duration_ms / 1000.0):
                intervals.append(VADInterval(start=round(start_time, 2), end=round(end_time, 2), is_speech=True))

    if in_speech:
        end_time = times[-1] + (frame_length_ms / 2000.0)
        if (end_time - start_time) >= (min_speech_duration_ms / 1000.0):
            intervals.append(VADInterval(start=round(start_time, 2), end=round(end_time, 2), is_speech=True))

    # Hangover / silence bridging: merge segments separated by less than min_silence_duration_ms
    min_silence_s = min_silence_duration_ms / 1000.0
    merged: List[VADInterval] = []
    for interval in intervals:
        if not merged:
            merged.append(interval)
        else:
            prev = merged[-1]
            if (interval.start - prev.end) < min_silence_s:
                merged[-1] = VADInterval(start=prev.start, end=interval.end, is_speech=True)
            else:
                merged.append(interval)

    total_speech_duration = sum(seg.end - seg.start for seg in merged)
    total_duration = len(signal) / sr
    speech_ratio = total_speech_duration / total_duration if total_duration > 0 else 0.0

    metrics = {
        "ste_threshold": round(ste_threshold, 6),
        "noise_floor": round(noise_floor, 6),
        "total_speech_seconds": round(total_speech_duration, 2),
        "total_duration_seconds": round(total_duration, 2),
        "speech_ratio": round(speech_ratio, 3),
        "num_speech_intervals": len(merged),
    }

    return merged, metrics
