"""Transcript integration engine: merges ASR segments with Diarization speaker turns.
Preserves segment ID, timestamps, text, speaker IDs, confidence, and word alignments.
"""

from typing import List
from src.core.models import TranscriptSegment, DiarizationSegment


def merge_transcript_and_diarization(
    transcript_segments: List[TranscriptSegment],
    diarization_segments: List[DiarizationSegment],
) -> List[TranscriptSegment]:
    """
    Assigns the most representative speaker to each transcript segment
    based on temporal overlap interval with diarization segments.
    """
    if not diarization_segments:
        return transcript_segments

    updated_segments: List[TranscriptSegment] = []

    for t_seg in transcript_segments:
        best_speaker = t_seg.speaker_id
        max_overlap = 0.0

        for d_seg in diarization_segments:
            # Calculate intersection [max(start1, start2), min(end1, end2)]
            overlap_start = max(t_seg.start, d_seg.start)
            overlap_end = min(t_seg.end, d_seg.end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = d_seg.speaker_id

        # If no overlap found (e.g. slight timing discrepancy), find closest diarization segment
        if max_overlap == 0.0 and diarization_segments:
            mid_t = (t_seg.start + t_seg.end) / 2.0
            closest = min(diarization_segments, key=lambda d: abs(((d.start + d.end) / 2.0) - mid_t))
            best_speaker = closest.speaker_id

        updated = TranscriptSegment(
            segment_id=t_seg.segment_id,
            start=t_seg.start,
            end=t_seg.end,
            text=t_seg.text,
            speaker_id=best_speaker,
            confidence=t_seg.confidence,
            alignment_status=t_seg.alignment_status,
            words=t_seg.words,
        )
        updated_segments.append(updated)

    return updated_segments
