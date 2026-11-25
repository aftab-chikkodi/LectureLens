"""Search and Information Retrieval Engine for Lecture Transcripts.
Provides:
- Case-insensitive keyword and exact phrase matching
- Speaker-based and timestamp-based range filtering
- Exact occurrence offsets for UI text highlighting
- Formatted passage contexts for targeted playback
"""

import re
from typing import List, Dict, Any, Optional
from src.core.models import TranscriptSegment


class SearchEngine:
    """Indexed search manager over lecture transcript segments."""

    def __init__(self, segments: List[TranscriptSegment]):
        self.segments = segments

    def search(
        self,
        query: str,
        speaker_filter: Optional[str] = None,
        min_start_time: Optional[float] = None,
        max_end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes query search with optional speaker and timestamp constraints.
        Returns matching segments with character spans and match metadata.
        """
        query = query.strip()
        if not query:
            return []

        results = []
        # Compile regex for case-insensitive matching with boundary flexibility
        pattern = re.compile(re.escape(query), re.IGNORECASE)

        for seg in self.segments:
            # Check speaker filter
            if speaker_filter and speaker_filter != "All Speakers":
                if seg.speaker_id != speaker_filter:
                    continue

            # Check timestamp bounds
            if min_start_time is not None and seg.end < min_start_time:
                continue
            if max_end_time is not None and seg.start > max_end_time:
                continue

            matches = list(pattern.finditer(seg.text))
            if matches:
                match_spans = [{"start": m.start(), "end": m.end()} for m in matches]
                results.append({
                    "segment_id": seg.segment_id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker_id": seg.speaker_id,
                    "match_count": len(matches),
                    "match_spans": match_spans,
                    "confidence": seg.confidence,
                })

        return results

    def get_speakers(self) -> List[str]:
        """Returns unique list of speaker identifiers present in the transcript."""
        speakers = sorted(list({seg.speaker_id for seg in self.segments}))
        return speakers
