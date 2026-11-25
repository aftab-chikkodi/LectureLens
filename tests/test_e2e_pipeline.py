"""Automated end-to-end integration test executing the complete LectureLens pipeline."""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.audio_io import validate_audio_file, load_audio
from src.core.preprocessing import preprocess_pipeline
from src.core.vad import detect_voice_activity
from src.core.diarization import diarize_audio
from src.core.transcript_merger import merge_transcript_and_diarization
from src.core.search import SearchEngine
from src.core.export import export_to_json, export_to_srt
from src.core.models import TranscriptSegment


def test_full_lecturelens_pipeline():
    audio_path = "data/samples/sample_lecture.wav"
    assert os.path.exists(audio_path), "Sample audio file must exist."

    # 1. Validation
    meta = validate_audio_file(audio_path)
    assert meta.duration_seconds > 0

    # 2. Loading & Preprocessing
    signal, sr = load_audio(audio_path, target_sr=16000, mono=True)
    norm_signal, stats = preprocess_pipeline(signal, sr=sr)
    assert np.max(np.abs(norm_signal)) <= 1.0

    # 3. Voice Activity Detection
    vad_intervals, vad_metrics = detect_voice_activity(norm_signal, sr=sr)
    assert len(vad_intervals) >= 1

    # 4. Speaker Diarization
    diar_segs, diar_meta = diarize_audio(norm_signal, sr=sr, vad_intervals=vad_intervals, num_speakers=2)
    assert len(diar_segs) >= 1

    # 5. Synthetic ASR Segments for end-to-end flow
    asr_segments = [
        TranscriptSegment(
            segment_id=1,
            start=0.5,
            end=4.0,
            text="Today we begin our discussion on speech signal processing and acoustic modeling.",
            confidence=0.96,
        ),
        TranscriptSegment(
            segment_id=2,
            start=6.2,
            end=11.2,
            text="Thank you professor. How does Dynamic Time Warping compare to Hidden Markov Models?",
            confidence=0.94,
        ),
    ]

    # 6. Integration & Merging
    merged_transcript = merge_transcript_and_diarization(asr_segments, diar_segs)
    assert len(merged_transcript) == 2
    assert merged_transcript[0].speaker_id != ""

    # 7. Search & Retrieval
    search_eng = SearchEngine(merged_transcript)
    dtw_matches = search_eng.search("Dynamic Time Warping")
    assert len(dtw_matches) == 1
    assert dtw_matches[0]["segment_id"] == 2

    # 8. Export
    json_export = export_to_json(merged_transcript)
    assert "Dynamic Time Warping" in json_export

    srt_export = export_to_srt(merged_transcript)
    assert "00:00:06,200 --> 00:00:11,200" in srt_export

    print("\n✅ End-to-End Pipeline Integration Test PASSED Successfully!")


if __name__ == "__main__":
    test_full_lecturelens_pipeline()
