"""Evaluation module for Speech Recognition (WER) and Speaker Diarization (DER).
Implements standard metrics without fabrication:
- Word Error Rate (WER) via Levenshtein distance
- Substitutions, Insertions, Deletions breakdown
- Diarization Error Rate estimation when ground truth is provided
"""

from typing import Dict, Any, List, Optional
import jiwer
from src.core.models import TranscriptSegment, DiarizationSegment


def calculate_wer(reference_text: str, hypothesis_text: str) -> Dict[str, Any]:
    """
    Computes exact Word Error Rate (WER) using Levenshtein distance.
    Returns WER, substitutions, deletions, insertions, and hit counts.
    """
    ref = reference_text.strip()
    hyp = hypothesis_text.strip()

    if not ref:
        return {"error": "Reference text is empty. Ground truth reference is required to calculate WER."}

    # Standard jiwer calculation
    output = jiwer.process_words(ref, hyp)

    return {
        "wer": round(output.wer, 4),
        "wer_percent": f"{output.wer * 100:.2f}%",
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "hits": output.hits,
        "reference_word_count": len(ref.split()),
        "hypothesis_word_count": len(hyp.split()),
    }


def calculate_der(
    ground_truth_diarization: List[DiarizationSegment],
    predicted_diarization: List[DiarizationSegment],
    collar: float = 0.25,
) -> Dict[str, Any]:
    """
    Computes Diarization Error Rate (DER) against ground truth segments:
    DER = (Missed Speech + False Alarm + Speaker Confusion) / Total Reference Speech Time
    """
    if not ground_truth_diarization:
        return {"error": "Ground truth speaker labels are missing. DER cannot be calculated without reference data."}

    # Sample time along 100ms grid to compute discrete overlaps accurately
    total_ref_duration = sum(max(0.0, s.end - s.start) for s in ground_truth_diarization)
    if total_ref_duration <= 0:
        return {"error": "Ground truth total duration is zero."}

    step = 0.1  # 100ms
    max_time = max(
        max(s.end for s in ground_truth_diarization),
        max(s.end for s in predicted_diarization) if predicted_diarization else 0.0,
    )

    times = [i * step for i in range(int(max_time / step) + 1)]
    missed_count = 0
    false_alarm_count = 0
    speaker_error_count = 0
    total_ref_speech_frames = 0

    for t in times:
        # Find active reference speaker
        ref_speaker = None
        for s in ground_truth_diarization:
            if s.start <= t <= s.end:
                ref_speaker = s.speaker_id
                break

        # Find active predicted speaker
        hyp_speaker = None
        for s in predicted_diarization:
            if s.start <= t <= s.end:
                hyp_speaker = s.speaker_id
                break

        if ref_speaker is not None:
            total_ref_speech_frames += 1
            if hyp_speaker is None:
                missed_count += 1
            elif hyp_speaker != ref_speaker:
                speaker_error_count += 1
        else:
            if hyp_speaker is not None:
                false_alarm_count += 1

    if total_ref_speech_frames == 0:
        return {"der": 0.0, "der_percent": "0.00%"}

    der_score = (missed_count + false_alarm_count + speaker_error_count) / float(total_ref_speech_frames)

    return {
        "der": round(der_score, 4),
        "der_percent": f"{der_score * 100:.2f}%",
        "missed_speech_ratio": round(missed_count / total_ref_speech_frames, 4),
        "false_alarm_ratio": round(false_alarm_count / total_ref_speech_frames, 4),
        "speaker_confusion_ratio": round(speaker_error_count / total_ref_speech_frames, 4),
    }
