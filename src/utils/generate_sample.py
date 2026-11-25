"""Generates synthetic multi-speaker speech-like signals for testing and verification."""

import numpy as np
import soundfile as sf
import os


def generate_synthetic_lecture(
    output_path: str = "data/samples/sample_lecture.wav",
    duration_sec: float = 12.0,
    sr: int = 16000,
) -> str:
    """
    Synthesizes a 12-second two-speaker audio recording with:
    - Speaker 1: Formant tones around 150 Hz pitch with harmonics (0s - 4.5s)
    - Silence pause: (4.5s - 6.0s)
    - Speaker 2: Formant tones around 240 Hz pitch with distinct harmonics (6.0s - 11.5s)
    - Silence pause: (11.5s - 12.0s)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_samples = int(duration_sec * sr)
    signal = np.zeros(total_samples, dtype=np.float32)

    # Speaker 1: Low fundamental pitch (150 Hz) + Formants (500Hz, 1500Hz, 2500Hz)
    t1 = np.linspace(0, 4.5, int(4.5 * sr), endpoint=False)
    spk1_sig = (
        0.5 * np.sin(2 * np.pi * 150 * t1)
        + 0.3 * np.sin(2 * np.pi * 500 * t1)
        + 0.2 * np.sin(2 * np.pi * 1500 * t1)
    )
    # Apply amplitude modulation to mimic syllabic cadence
    cadence1 = 0.5 * (1 + np.sin(2 * np.pi * 3.5 * t1))
    spk1_sig = spk1_sig * cadence1

    # Insert Speaker 1 into 0s - 4.5s
    signal[0 : len(spk1_sig)] = spk1_sig

    # Speaker 2: Higher pitch (240 Hz) + Formants (800Hz, 2000Hz, 3200Hz)
    t2 = np.linspace(0, 5.5, int(5.5 * sr), endpoint=False)
    spk2_sig = (
        0.5 * np.sin(2 * np.pi * 240 * t2)
        + 0.3 * np.sin(2 * np.pi * 800 * t2)
        + 0.2 * np.sin(2 * np.pi * 2000 * t2)
    )
    cadence2 = 0.5 * (1 + np.sin(2 * np.pi * 4.0 * t2))
    spk2_sig = spk2_sig * cadence2

    start_spk2 = int(6.0 * sr)
    signal[start_spk2 : start_spk2 + len(spk2_sig)] = spk2_sig

    # Add light background noise
    noise = 0.005 * np.random.randn(total_samples)
    final_signal = (signal + noise).astype(np.float32)

    # Normalize
    final_signal = final_signal / np.max(np.abs(final_signal)) * 0.9

    sf.write(output_path, final_signal, sr, subtype="PCM_16")
    return output_path


if __name__ == "__main__":
    path = generate_synthetic_lecture()
    print(f"Sample lecture audio generated at: {path}")
