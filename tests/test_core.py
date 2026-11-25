"""Comprehensive automated test suite for LectureLens core modules."""

import os
import pytest
import numpy as np
from src.core.models import TranscriptSegment, DiarizationSegment, VADInterval
from src.core.audio_io import validate_audio_file, load_audio, AudioValidationError
from src.core.preprocessing import preprocess_pipeline, to_mono, normalize_peak
from src.core.vad import detect_voice_activity
from src.core.diarization import diarize_audio
from src.core.transcript_merger import merge_transcript_and_diarization
from src.core.search import SearchEngine
from src.core.evaluation import calculate_wer, calculate_der
from src.core.export import export_to_txt, export_to_json, export_to_csv, export_to_srt
from src.core.spr_algorithms import dynamic_time_warping, viterbi_decoding_demo, compute_lpc_levinson


SAMPLE_AUDIO = "data/samples/sample_lecture.wav"


def test_audio_validation_success():
    meta = validate_audio_file(SAMPLE_AUDIO)
    assert meta.duration_seconds > 5.0
    assert meta.sample_rate == 16000
    assert meta.channels == 1


def test_audio_validation_failure():
    with pytest.raises(AudioValidationError):
        validate_audio_file("non_existent_file.xyz")


def test_audio_loading_and_preprocessing():
    sig, sr = load_audio(SAMPLE_AUDIO, target_sr=16000)
    assert len(sig) > 0
    assert sr == 16000

    proc_sig, stats = preprocess_pipeline(sig, sr=sr)
    assert np.max(np.abs(proc_sig)) <= 1.0
    assert "estimated_snr_db" in stats


def test_voice_activity_detection():
    sig, sr = load_audio(SAMPLE_AUDIO, target_sr=16000)
    intervals, metrics = detect_voice_activity(sig, sr=sr)
    assert len(intervals) >= 1
    assert metrics["speech_ratio"] > 0.0


def test_speaker_diarization():
    sig, sr = load_audio(SAMPLE_AUDIO, target_sr=16000)
    intervals, _ = detect_voice_activity(sig, sr=sr)
    diar_segs, meta = diarize_audio(sig, sr=sr, vad_intervals=intervals, num_speakers=2)
    assert len(diar_segs) >= 1
    assert any("Speaker" in s.speaker_id for s in diar_segs)


def test_transcript_merging():
    t_segs = [
        TranscriptSegment(segment_id=1, start=0.5, end=3.5, text="Welcome to speech processing lecture"),
        TranscriptSegment(segment_id=2, start=6.5, end=10.0, text="Thank you professor for the introduction"),
    ]
    d_segs = [
        DiarizationSegment(start=0.0, end=4.5, speaker_id="Speaker 1"),
        DiarizationSegment(start=6.0, end=11.5, speaker_id="Speaker 2"),
    ]
    merged = merge_transcript_and_diarization(t_segs, d_segs)
    assert merged[0].speaker_id == "Speaker 1"
    assert merged[1].speaker_id == "Speaker 2"


def test_search_engine():
    segs = [
        TranscriptSegment(segment_id=1, start=0.0, end=3.0, text="Dynamic Time Warping is used for alignment", speaker_id="Speaker 1"),
        TranscriptSegment(segment_id=2, start=3.5, end=6.0, text="Linear Predictive Coding models vocal tract", speaker_id="Speaker 2"),
    ]
    engine = SearchEngine(segs)
    res = engine.search("warping")
    assert len(res) == 1
    assert res[0]["segment_id"] == 1
    assert res[0]["match_count"] == 1

    # Filter test
    res_filtered = engine.search("vocal", speaker_filter="Speaker 1")
    assert len(res_filtered) == 0


def test_spr_algorithms():
    # 1. DTW
    seq1 = np.array([1, 2, 3, 4, 5])
    seq2 = np.array([1, 1, 2, 3, 4, 5, 5])
    dist, path, _ = dynamic_time_warping(seq1, seq2)
    assert dist >= 0.0
    assert len(path) > 0

    # 2. LPC Levinson-Durbin
    frame = np.sin(np.linspace(0, 10, 160))
    lpc, err = compute_lpc_levinson(frame, order=10)
    assert len(lpc) == 11
    assert lpc[0] == 1.0

    # 3. HMM Viterbi
    states = ["S0", "S1"]
    obs = [0, 1, 0]
    pi = np.array([0.5, 0.5])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.9, 0.1], [0.2, 0.8]])
    best_path, prob, _ = viterbi_decoding_demo(obs, states, pi, A, B)
    assert len(best_path) == 3


def test_evaluation_wer_and_der():
    ref = "the quick brown fox jumps over the lazy dog"
    hyp = "the quick brown fox jumped over a lazy dog"
    wer_res = calculate_wer(ref, hyp)
    assert wer_res["wer"] > 0.0
    assert wer_res["substitutions"] == 2

    gt_diar = [DiarizationSegment(start=0.0, end=5.0, speaker_id="Speaker 1")]
    pred_diar = [DiarizationSegment(start=0.0, end=5.0, speaker_id="Speaker 1")]
    der_res = calculate_der(gt_diar, pred_diar)
    assert der_res["der"] == 0.0


def test_export_formats():
    segs = [
        TranscriptSegment(segment_id=1, start=1.0, end=4.5, text="Testing LectureLens export functions", speaker_id="Speaker 1")
    ]
    txt = export_to_txt(segs)
    assert "Testing LectureLens" in txt
    assert "[1.00s - 4.50s]" in txt

    json_str = export_to_json(segs)
    assert '"segment_id": 1' in json_str

    csv_str = export_to_csv(segs)
    assert "Segment ID" in csv_str

    srt_str = export_to_srt(segs)
    assert "00:00:01,000 --> 00:00:04,500" in srt_str
