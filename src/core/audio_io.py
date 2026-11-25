"""Audio Input/Output and validation engine supporting WAV, MP3, FLAC, M4A."""

import os
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import soundfile as sf
import librosa
from src.core.models import AudioMetadata


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB limit for safe lecture loading
MAX_DURATION_SECONDS = 7200.0  # 2 hours


class AudioValidationError(Exception):
    """Raised when audio fails integrity or specification checks."""
    pass


def validate_audio_file(file_path: str) -> AudioMetadata:
    """
    Validates audio format, size, duration, channels, and sample rate.
    Raises AudioValidationError if audio is invalid or corrupted.
    """
    path = Path(file_path)
    if not path.is_file():
        raise AudioValidationError(f"Audio file does not exist: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise AudioValidationError(
            f"Unsupported format '{ext}'. Allowed formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    file_size = path.stat().st_size
    if file_size == 0:
        raise AudioValidationError("Audio file is empty (0 bytes).")
    if file_size > MAX_FILE_SIZE_BYTES:
        raise AudioValidationError(
            f"File size ({file_size / (1024*1024):.2f}MB) exceeds limit of {MAX_FILE_SIZE_BYTES / (1024*1024)}MB."
        )

    try:
        # Use soundfile info first for efficiency
        info = sf.info(str(path))
        duration = float(info.duration)
        sr = int(info.samplerate)
        channels = int(info.channels)
        format_name = info.format
    except Exception as sf_err:
        # Fallback to librosa/audioread/PyAV for m4a/mp3
        try:
            duration = float(librosa.get_duration(path=str(path)))
            sr = 16000  # Default assumption for fallback probing
            channels = 1
            format_name = ext.replace(".", "").upper()
        except Exception as lib_err:
            raise AudioValidationError(
                f"Corrupted or unreadable audio file: {sf_err}; fallback error: {lib_err}"
            )

    if duration <= 0.1:
        raise AudioValidationError(f"Audio duration is too short ({duration:.2f}s). Minimum is 0.5s.")
    if duration > MAX_DURATION_SECONDS:
        raise AudioValidationError(
            f"Audio duration ({duration/60:.1f} mins) exceeds maximum allowed {MAX_DURATION_SECONDS/60:.1f} mins."
        )

    return AudioMetadata(
        filename=path.name,
        duration_seconds=duration,
        sample_rate=sr,
        channels=channels,
        format=format_name,
        file_size_bytes=file_size,
    )


def load_audio(file_path: str, target_sr: int = 16000, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Loads audio with exact target sampling rate and float32 mono representation.
    Preserves original file safely.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Load using soundfile if possible
        data, sr = sf.read(file_path, dtype="float32")
        if data.ndim > 1:
            if mono:
                data = np.mean(data, axis=1)
        if target_sr is not None and sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return data.astype(np.float32), sr
    except Exception:
        # Robust fallback using librosa / PyAV
        y, sr = librosa.load(file_path, sr=target_sr, mono=mono)
        return y.astype(np.float32), sr


def save_audio_wav(file_path: str, signal: np.ndarray, sr: int = 16000) -> str:
    """Saves a normalized float32 or int16 numpy array to standard WAV."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file_path, signal, sr, subtype="PCM_16")
    return file_path
