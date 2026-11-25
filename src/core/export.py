"""Export utilities for transcripts in TXT, JSON, CSV, and SRT subtitle formats."""

import json
import csv
import io
from typing import List
from src.core.models import TranscriptSegment


def format_timestamp_srt(seconds: float) -> str:
    """Formats seconds into standard SRT timestamp format: HH:MM:SS,mmm"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def export_to_txt(segments: List[TranscriptSegment]) -> str:
    """Exports transcript to clean readable text format."""
    lines = []
    for s in segments:
        lines.append(f"[{s.start:.2f}s - {s.end:.2f}s] {s.speaker_id}: {s.text}")
    return "\n".join(lines)


def export_to_json(segments: List[TranscriptSegment]) -> str:
    """Exports full structured transcript metadata to JSON."""
    data = [s.to_dict() for s in segments]
    return json.dumps(data, indent=2, ensure_ascii=False)


def export_to_csv(segments: List[TranscriptSegment]) -> str:
    """Exports transcript to CSV format."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Segment ID", "Start (s)", "End (s)", "Speaker", "Text", "Confidence", "Alignment Status"])
    for s in segments:
        writer.writerow([
            s.segment_id,
            f"{s.start:.2f}",
            f"{s.end:.2f}",
            s.speaker_id,
            s.text,
            f"{s.confidence:.3f}" if s.confidence is not None else "N/A",
            s.alignment_status,
        ])
    return output.getvalue()


def export_to_srt(segments: List[TranscriptSegment]) -> str:
    """Exports transcript to standard SRT subtitle file."""
    srt_blocks = []
    for idx, s in enumerate(segments, start=1):
        start_str = format_timestamp_srt(s.start)
        end_str = format_timestamp_srt(s.end)
        text_line = f"[{s.speaker_id}] {s.text}"
        srt_blocks.append(f"{idx}\n{start_str} --> {end_str}\n{text_line}\n")
    return "\n".join(srt_blocks)
