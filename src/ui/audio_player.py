"""Custom Interactive HTML5/JavaScript Audio Player and Synchronized Transcript Component.
Ensures the client-side audio element is the single source of truth for:
- Play, Pause, Restart, Seek -10s, Seek +10s
- Variable Playback Speeds: 0.75x, 1x, 1.25x, 1.5x, 2x
- Volume slider & Mute toggle
- Waveform timeline seek bar
- Clickable transcript segments with auto-scroll and dynamic active segment highlighting
- Clickable search result navigation and passage playback with boundaries
"""

import base64
import json
from typing import List, Dict, Any, Optional
from src.core.models import TranscriptSegment


def render_audio_player_component(
    audio_bytes: bytes,
    audio_format: str,
    segments: List[TranscriptSegment],
    search_results: Optional[List[Dict[str, Any]]] = None,
    initial_seek_time: float = 0.0,
    speaker_colors: Optional[Dict[str, str]] = None,
) -> str:
    """
    Renders an embedded, self-contained HTML5/JS audio player with direct
    DOM-level synchronization to transcript segments and search matches.
    """
    if speaker_colors is None:
        speaker_colors = {
            "Speaker 1": "#2563EB",
            "Speaker 2": "#D97706",
            "Speaker 3": "#7C3AED",
            "Speaker 4": "#DB2777",
            "Speaker 5": "#0D9488",
        }

    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    mime_type = "audio/wav" if "wav" in audio_format.lower() else "audio/mpeg"
    audio_src = f"data:{mime_type};base64,{b64_audio}"

    segments_data = [s.to_dict() for s in segments]
    segments_json = json.dumps(segments_data)
    search_json = json.dumps(search_results or [])

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <style>
        :root {{
          --primary: #2563EB;
          --primary-hover: #1D4ED8;
          --bg-card: #FFFFFF;
          --text-main: #1E293B;
          --text-muted: #64748B;
          --border: #E2E8F0;
          --highlight: #FEF08A;
          --active-segment: #EFF6FF;
          --active-border: #3B82F6;
        }}
        body {{
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
          margin: 0;
          padding: 8px;
          background: transparent;
          color: var(--text-main);
        }}
        .player-container {{
          background: #FFFFFF;
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 18px;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
          margin-bottom: 16px;
        }}
        .controls-row {{
          display: flex;
          align-items: center;
          gap: 12px;
          flex-wrap: wrap;
          margin-top: 12px;
        }}
        .btn {{
          background: #F1F5F9;
          border: 1px solid var(--border);
          border-radius: 6px;
          padding: 8px 14px;
          cursor: pointer;
          font-size: 13px;
          font-weight: 600;
          color: var(--text-main);
          transition: all 0.15s ease;
          display: inline-flex;
          align-items: center;
          gap: 4px;
        }}
        .btn:hover {{
          background: #E2E8F0;
        }}
        .btn-primary {{
          background: var(--primary);
          color: #FFFFFF;
          border-color: var(--primary);
        }}
        .btn-primary:hover {{
          background: var(--primary-hover);
        }}
        .timeline-container {{
          width: 100%;
          margin: 12px 0 6px 0;
        }}
        .timeline-slider {{
          width: 100%;
          cursor: pointer;
          accent-color: var(--primary);
        }}
        .time-display {{
          font-size: 13px;
          font-weight: 500;
          color: var(--text-muted);
          min-width: 110px;
        }}
        .rate-select, .volume-slider {{
          padding: 6px 10px;
          border-radius: 6px;
          border: 1px solid var(--border);
          font-size: 13px;
          background: #FFFFFF;
        }}
        .status-badge {{
          font-size: 12px;
          padding: 4px 8px;
          border-radius: 4px;
          background: #F1F5F9;
          color: #475569;
          font-weight: 500;
        }}
        /* Transcript container */
        .transcript-container {{
          max-height: 480px;
          overflow-y: auto;
          border: 1px solid var(--border);
          border-radius: 10px;
          background: #FAFAFA;
          padding: 12px;
        }}
        .segment-card {{
          background: #FFFFFF;
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 12px 14px;
          margin-bottom: 10px;
          cursor: pointer;
          transition: all 0.15s ease;
          border-left: 4px solid #94A3B8;
        }}
        .segment-card:hover {{
          border-color: #CBD5E1;
          box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        }}
        .segment-card.active {{
          background: var(--active-segment);
          border-color: var(--active-border);
          box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
        }}
        .segment-header {{
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 6px;
          font-size: 12px;
        }}
        .speaker-tag {{
          font-weight: 700;
          padding: 2px 8px;
          border-radius: 4px;
          color: #FFFFFF;
          font-size: 11px;
        }}
        .timestamp-tag {{
          color: var(--text-muted);
          font-family: monospace;
          font-size: 12px;
        }}
        .segment-text {{
          font-size: 14px;
          line-height: 1.5;
          color: #334155;
        }}
        .search-match {{
          background: var(--highlight);
          font-weight: 600;
          padding: 1px 3px;
          border-radius: 2px;
        }}
      </style>
    </head>
    <body>

    <div class="player-container">
      <audio id="lectureAudio" preload="auto">
        <source src="{audio_src}" type="{mime_type}">
        Your browser does not support the audio element.
      </audio>

      <!-- Waveform Progress Timeline -->
      <div class="timeline-container">
        <input type="range" id="timelineSlider" class="timeline-slider" min="0" max="100" value="0" step="0.1">
      </div>

      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="time-display" id="timeDisplay">0:00 / 0:00</div>
        <div id="passageNotice" class="status-badge" style="display: none; background: #FEF3C7; color: #92400E;">
          Playing Selected Passage
        </div>
      </div>

      <!-- Player Controls -->
      <div class="controls-row">
        <button id="btnPlayPause" class="btn btn-primary" onclick="togglePlay()">Play</button>
        <button class="btn" onclick="seekRelative(-10)">-10s</button>
        <button class="btn" onclick="seekRelative(10)">+10s</button>
        <button class="btn" onclick="restartAudio()">Restart</button>

        <label style="font-size: 13px; font-weight: 500; margin-left: 8px;">Speed:</label>
        <select id="rateSelect" class="rate-select" onchange="changePlaybackRate(this.value)">
          <option value="0.75">0.75x</option>
          <option value="1.0" selected>1.0x</option>
          <option value="1.25">1.25x</option>
          <option value="1.5">1.5x</option>
          <option value="2.0">2.0x</option>
        </select>

        <label style="font-size: 13px; font-weight: 500; margin-left: 8px;">Vol:</label>
        <input type="range" id="volumeSlider" class="volume-slider" min="0" max="1" step="0.05" value="1" oninput="changeVolume(this.value)" style="width: 80px;">
        <button id="btnMute" class="btn" onclick="toggleMute()">Mute</button>
      </div>
    </div>

    <!-- Clickable Interactive Transcript List -->
    <div style="margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
      <h4 style="margin: 0; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted);">
        Interactive Synchronized Transcript
      </h4>
      <span style="font-size: 12px; color: var(--text-muted);">Click any segment to jump and play</span>
    </div>

    <div class="transcript-container" id="transcriptList"></div>

    <script>
      const audio = document.getElementById('lectureAudio');
      const btnPlayPause = document.getElementById('btnPlayPause');
      const timelineSlider = document.getElementById('timelineSlider');
      const timeDisplay = document.getElementById('timeDisplay');
      const rateSelect = document.getElementById('rateSelect');
      const volumeSlider = document.getElementById('volumeSlider');
      const btnMute = document.getElementById('btnMute');
      const transcriptList = document.getElementById('transcriptList');
      const passageNotice = document.getElementById('passageNotice');

      const segments = {segments_json};
      const searchResults = {search_json};
      const speakerColors = {json.dumps(speaker_colors)};

      let passageEnd = null;
      let activeSegmentId = null;

      function formatTime(secs) {{
        if (isNaN(secs)) return "0:00";
        const m = Math.floor(secs / 60);
        const s = Math.floor(secs % 60);
        return m + ":" + (s < 10 ? "0" : "") + s;
      }}

      function updateUI() {{
        const cur = audio.currentTime;
        const dur = audio.duration || 0;
        timelineSlider.value = dur > 0 ? (cur / dur) * 100 : 0;
        timeDisplay.innerText = formatTime(cur) + " / " + formatTime(dur);

        // Check passage boundary
        if (passageEnd !== null && cur >= passageEnd) {{
          audio.pause();
          passageEnd = null;
          passageNotice.style.display = "none";
          btnPlayPause.innerText = "Play";
        }}

        // Synchronize active transcript segment
        let currentSeg = null;
        for (let i = 0; i < segments.length; i++) {{
          if (cur >= segments[i].start && cur <= segments[i].end) {{
            currentSeg = segments[i];
            break;
          }}
        }}

        if (currentSeg && currentSeg.segment_id !== activeSegmentId) {{
          if (activeSegmentId) {{
            const prevEl = document.getElementById('seg-' + activeSegmentId);
            if (prevEl) prevEl.classList.remove('active');
          }}
          activeSegmentId = currentSeg.segment_id;
          const newEl = document.getElementById('seg-' + activeSegmentId);
          if (newEl) {{
            newEl.classList.add('active');
            newEl.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
          }}
        }}
      }}

      function togglePlay() {{
        if (audio.paused) {{
          audio.play().then(() => {{
            btnPlayPause.innerText = "Pause";
          }}).catch(err => console.log("Playback error: ", err));
        }} else {{
          audio.pause();
          btnPlayPause.innerText = "Play";
        }}
      }}

      function seekRelative(offset) {{
        audio.currentTime = Math.max(0, Math.min(audio.duration || 0, audio.currentTime + offset));
      }}

      function restartAudio() {{
        audio.currentTime = 0;
        audio.play();
        btnPlayPause.innerText = "Pause";
      }}

      function changePlaybackRate(val) {{
        audio.playbackRate = parseFloat(val);
      }}

      function changeVolume(val) {{
        audio.volume = parseFloat(val);
        audio.muted = (val === "0");
        btnMute.innerText = audio.muted ? "Unmute" : "Mute";
      }}

      function toggleMute() {{
        audio.muted = !audio.muted;
        btnMute.innerText = audio.muted ? "Unmute" : "Mute";
      }}

      function seekToSegment(start, end, isPassage = false) {{
        passageEnd = isPassage ? end : null;
        passageNotice.style.display = isPassage ? "inline-block" : "none";
        audio.currentTime = start;
        audio.play().then(() => {{
          btnPlayPause.innerText = "Pause";
        }}).catch(e => console.log(e));
      }}

      timelineSlider.addEventListener('input', (e) => {{
        if (audio.duration) {{
          audio.currentTime = (e.target.value / 100) * audio.duration;
        }}
      }});

      audio.addEventListener('timeupdate', updateUI);
      audio.addEventListener('loadedmetadata', updateUI);
      audio.addEventListener('play', () => btnPlayPause.innerText = "Pause");
      audio.addEventListener('pause', () => btnPlayPause.innerText = "Play");

      // Render Transcript Cards
      function renderTranscript() {{
        transcriptList.innerHTML = "";
        segments.forEach(seg => {{
          const card = document.createElement('div');
          card.className = "segment-card";
          card.id = "seg-" + seg.segment_id;
          const color = speakerColors[seg.speaker_id] || "#64748B";
          card.style.borderLeftColor = color;

          let highlightedText = seg.text;
          // Check if this segment has search matches
          const matchObj = searchResults.find(r => r.segment_id === seg.segment_id);
          if (matchObj && matchObj.match_spans) {{
            // Highlight matching substrings
            let newText = "";
            let lastIdx = 0;
            matchObj.match_spans.forEach(span => {{
              newText += seg.text.substring(lastIdx, span.start);
              newText += '<span class="search-match">' + seg.text.substring(span.start, span.end) + '</span>';
              lastIdx = span.end;
            }});
            newText += seg.text.substring(lastIdx);
            highlightedText = newText;
          }}

          card.innerHTML = `
            <div class="segment-header">
              <span class="speaker-tag" style="background: ${{color}}">${{seg.speaker_id}}</span>
              <span class="timestamp-tag">${{formatTime(seg.start)}} - ${{formatTime(seg.end)}}</span>
            </div>
            <div class="segment-text">${{highlightedText}}</div>
          `;

          card.onclick = () => seekToSegment(seg.start, seg.end, false);
          transcriptList.appendChild(card);
        }});
      }}

      renderTranscript();

      // Initial seek position
      if ({initial_seek_time} > 0) {{
        audio.addEventListener('canplay', () => {{
          audio.currentTime = {initial_seek_time};
        }}, {{ once: true }});
      }}
    </script>
    </body>
    </html>
    """
    return html
