"""Data models for LectureLens transcripts, speakers, and alignment metadata."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class WordAlignment:
    """Word-level alignment details."""
    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class TranscriptSegment:
    """Standardized transcript segment representation."""
    segment_id: int
    start: float
    end: float
    text: str
    speaker_id: str = "Speaker 1"
    confidence: Optional[float] = None
    alignment_status: str = "segment_level"  # 'word_level', 'segment_level', 'approximate'
    words: List[WordAlignment] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptSegment":
        words_data = data.get("words", [])
        words = [WordAlignment(**w) if isinstance(w, dict) else w for w in words_data]
        return cls(
            segment_id=data["segment_id"],
            start=float(data["start"]),
            end=float(data["end"]),
            text=str(data["text"]).strip(),
            speaker_id=str(data.get("speaker_id", "Speaker 1")),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            alignment_status=str(data.get("alignment_status", "segment_level")),
            words=words,
        )


@dataclass
class AudioMetadata:
    """Metadata regarding an uploaded or processed audio recording."""
    filename: str
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str
    file_size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiarizationSegment:
    """Speaker turn detection output."""
    start: float
    end: float
    speaker_id: str
    confidence: float = 1.0


@dataclass
class VADInterval:
    """Voice Activity Detection interval."""
    start: float
    end: float
    is_speech: bool = True
