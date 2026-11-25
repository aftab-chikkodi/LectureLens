"""LectureLens: Intelligent Lecture Navigation & Speaker-Aware Search.
Main Streamlit Application Entrypoint.
Course: MSc AI & Machine Learning - Speech Processing and Recognition (SPR).
"""

import os
import sys
import time
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.models import AudioMetadata, TranscriptSegment, DiarizationSegment, VADInterval
from src.core.audio_io import validate_audio_file, load_audio, save_audio_wav, AudioValidationError
from src.core.preprocessing import preprocess_pipeline
from src.core.vad import detect_voice_activity
from src.core.diarization import diarize_audio
from src.core.asr import ASREngine
from src.core.transcript_merger import merge_transcript_and_diarization
from src.core.search import SearchEngine
from src.core.visualizations import (
    plot_waveform,
    plot_spectrogram,
    plot_mel_spectrogram,
    plot_mfcc,
    plot_lpc_spectrum,
    plot_vad_and_diarization_timeline,
)
from src.core.evaluation import calculate_wer, calculate_der
from src.core.export import export_to_txt, export_to_json, export_to_csv, export_to_srt
from src.core.spr_algorithms import dynamic_time_warping, viterbi_decoding_demo
from src.ui.audio_player import render_audio_player_component


# Page configuration
st.set_page_config(
    page_title="LectureLens - Intelligent Lecture Navigation",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "processed_audio" not in st.session_state:
    st.session_state.processed_audio = None
if "sample_rate" not in st.session_state:
    st.session_state.sample_rate = 16000
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "vad_intervals" not in st.session_state:
    st.session_state.vad_intervals = []
if "vad_metrics" not in st.session_state:
    st.session_state.vad_metrics = {}
if "diarization_segments" not in st.session_state:
    st.session_state.diarization_segments = []
if "transcript_segments" not in st.session_state:
    st.session_state.transcript_segments = []
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Audio Player & Transcript"
if "asr_model_size" not in st.session_state:
    st.session_state.asr_model_size = "tiny"


# Sidebar navigation
st.sidebar.title("🎙️ LectureLens")
st.sidebar.caption("Intelligent Lecture Navigation & Speaker Search")
st.sidebar.markdown("---")

navigation = st.sidebar.radio(
    "Navigation",
    [
        "1. Upload & Process",
        "2. Audio Player & Transcript",
        "3. Speaker & Search Navigation",
        "4. Signal Visualizations",
        "5. SPR Course Algorithms (Viva)",
        "6. Evaluation & Benchmarking",
        "7. Data Export",
        "8. Technical Documentation",
    ],
)

# Status pill
if st.session_state.audio_path:
    dur = st.session_state.metadata.duration_seconds if st.session_state.metadata else 0
    st.sidebar.success(f"Loaded: {os.path.basename(st.session_state.audio_path)} ({dur:.1f}s)")
else:
    st.sidebar.info("No audio loaded. Upload a recording to begin.")


# -------------------------------------------------------------
# TAB 1: UPLOAD & PROCESS
# -------------------------------------------------------------
if navigation == "1. Upload & Process":
    st.header("📤 Lecture Audio Upload & Full Pipeline Processing")
    st.markdown(
        "Upload a lecture recording (WAV, MP3, FLAC, M4A). The pipeline validates audio integrity, performs "
        "time-frequency preprocessing, Voice Activity Detection, local Whisper ASR, and Speaker Diarization."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Select Audio File",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            help="Maximum file size: 100MB.",
        )

        st.markdown("#### ASR Model Configuration")
        model_size = st.selectbox(
            "Whisper Quantized Model Size (Offline CPU)",
            options=["tiny", "base", "small"],
            index=0,
            help="'tiny' offers fast real-time transcription on local CPU with int8 quantization.",
        )
        st.session_state.asr_model_size = model_size

        num_speakers_opt = st.slider(
            "Expected Number of Speakers (0 = Automatic Clustering)",
            min_value=0,
            max_value=5,
            value=0,
            help="Set to 0 to let Agglomerative Clustering dynamically estimate speaker clusters via silhouette score.",
        )

    with col2:
        st.info("ℹ️ **Standard Pipeline Stages:**\n"
                "1. Format & Duration Validation\n"
                "2. 16kHz Resampling & Peak Normalization\n"
                "3. Short-Time Energy VAD\n"
                "4. Offline Faster-Whisper ASR\n"
                "5. MFCC-Delta Speaker Diarization\n"
                "6. Temporal Segment Integration")

    if uploaded_file is not None:
        save_dir = os.path.join(PROJECT_ROOT, "data", "uploads")
        os.makedirs(save_dir, exist_ok=True)
        saved_file_path = os.path.join(save_dir, uploaded_file.name)

        with open(saved_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # 1. Validation
            metadata = validate_audio_file(saved_file_path)
            st.session_state.metadata = metadata
            st.session_state.audio_path = saved_file_path

            st.success(f"File validated successfully! Format: {metadata.format}, Duration: {metadata.duration_seconds:.2f}s, Channels: {metadata.channels}, SR: {metadata.sample_rate}Hz")

            if st.button("🚀 Run Full LectureLens Pipeline", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Step 1: Loading
                status_text.text("1/5 Loading & Resampling to 16kHz...")
                raw_signal, sr = load_audio(saved_file_path, target_sr=16000, mono=True)
                progress_bar.progress(20)

                # Step 2: Preprocessing
                status_text.text("2/5 Performing Peak Normalization & Noise Estimation...")
                proc_signal, noise_stats = preprocess_pipeline(raw_signal, sr=sr)
                st.session_state.processed_audio = proc_signal
                st.session_state.sample_rate = sr
                progress_bar.progress(40)

                # Step 3: VAD
                status_text.text("3/5 Computing Short-Time Energy & Zero-Crossing VAD...")
                vad_intervals, vad_metrics = detect_voice_activity(proc_signal, sr=sr)
                st.session_state.vad_intervals = vad_intervals
                st.session_state.vad_metrics = vad_metrics
                progress_bar.progress(60)

                # Step 4: ASR
                status_text.text(f"4/5 Transcribing with local Faster-Whisper ({model_size})...")
                asr_engine = ASREngine(model_size=model_size)
                # Pass normalized signal
                t_segs, asr_meta = asr_engine.transcribe(proc_signal, sr=sr, word_timestamps=True)
                progress_bar.progress(80)

                # Step 5: Diarization & Merge
                status_text.text("5/5 Diarizing Speakers (MFCC + Agglomerative Clustering) & Merging...")
                n_spk = None if num_speakers_opt == 0 else num_speakers_opt
                diar_segs, diar_meta = diarize_audio(proc_signal, sr=sr, vad_intervals=vad_intervals, num_speakers=n_spk)
                st.session_state.diarization_segments = diar_segs

                merged_segments = merge_transcript_and_diarization(t_segs, diar_segs)
                st.session_state.transcript_segments = merged_segments
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")

                st.balloons()
                st.success(f"Generated {len(merged_segments)} timestamped transcript segments across {diar_meta.get('num_speakers', 1)} speakers!")

        except AudioValidationError as err:
            st.error(f"Validation Failed: {err}")
        except Exception as err:
            st.error(f"Processing Error: {err}")


# -------------------------------------------------------------
# TAB 2: AUDIO PLAYER & TRANSCRIPT
# -------------------------------------------------------------
elif navigation == "2. Audio Player & Transcript":
    st.header("🎧 Synchronized Audio Player & Transcript")
    st.markdown(
        "Interactive playback client. **Click any transcript segment** to jump playback directly to that position. "
        "The transcript automatically highlights the active sentence in real time."
    )

    if not st.session_state.audio_path or not os.path.exists(st.session_state.audio_path):
        st.warning("Please upload and process an audio file in 'Upload & Process' first.")
    else:
        with open(st.session_state.audio_path, "rb") as f:
            audio_bytes = f.read()

        ext = Path(st.session_state.audio_path).suffix.replace(".", "")
        player_html = render_audio_player_component(
            audio_bytes=audio_bytes,
            audio_format=ext,
            segments=st.session_state.transcript_segments,
            initial_seek_time=0.0,
        )
        components.html(player_html, height=650, scrolling=True)


# -------------------------------------------------------------
# TAB 3: SPEAKER & SEARCH NAVIGATION
# -------------------------------------------------------------
elif navigation == "3. Speaker & Search Navigation":
    st.header("🔍 Intelligent Search & Speaker Navigation")
    st.markdown("Perform keyword or phrase queries with speaker filtering. Jump directly to matching moments in the audio.")

    if not st.session_state.transcript_segments:
        st.warning("No transcript available. Upload and process an audio recording first.")
    else:
        search_engine = SearchEngine(st.session_state.transcript_segments)
        available_speakers = ["All Speakers"] + search_engine.get_speakers()

        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            query = st.text_input("Enter search phrase or keyword:", placeholder="e.g. speech, transform, model...")
        with c2:
            selected_speaker = st.selectbox("Filter by Speaker", options=available_speakers)
        with c3:
            st.write("")
            st.write("")
            search_btn = st.button("Search", type="primary")

        if query:
            results = search_engine.search(query=query, speaker_filter=selected_speaker)
            st.write(f"**Found {len(results)} matching segment(s)**")

            if results:
                # Render results with clickable navigation
                with open(st.session_state.audio_path, "rb") as f:
                    audio_bytes = f.read()
                ext = Path(st.session_state.audio_path).suffix.replace(".", "")

                st.markdown("#### Audio Playback with Search Highlights")
                player_html = render_audio_player_component(
                    audio_bytes=audio_bytes,
                    audio_format=ext,
                    segments=st.session_state.transcript_segments,
                    search_results=results,
                )
                components.html(player_html, height=550, scrolling=True)

                st.markdown("#### Matching Segments List")
                for r in results:
                    with st.expander(f"⏱️ [{r['start']:.2f}s - {r['end']:.2f}s] {r['speaker_id']}: {r['match_count']} match(es)"):
                        st.markdown(f"**Text:** {r['text']}")
                        st.caption(f"Segment ID: {r['segment_id']} | Confidence: {r['confidence']}")
            else:
                st.info("No transcript segments matched your query.")


# -------------------------------------------------------------
# TAB 4: SIGNAL VISUALIZATIONS
# -------------------------------------------------------------
elif navigation == "4. Signal Visualizations":
    st.header("📊 Speech Signal Processing Visualizations")
    st.markdown("Visual representations generated directly from your uploaded lecture audio.")

    if st.session_state.processed_audio is None:
        st.warning("No audio loaded. Upload and process an audio file to view visualizations.")
    else:
        sig = st.session_state.processed_audio
        sr = st.session_state.sample_rate

        viz_type = st.radio(
            "Select Visualization",
            [
                "Waveform (Time Domain)",
                "STFT Spectrogram (Frequency Domain)",
                "Mel-Scale Spectrogram",
                "MFCC Heatmap",
                "LPC Spectral Envelope & Formants",
                "VAD & Speaker Diarization Alignment",
            ],
            horizontal=True,
        )

        if viz_type == "Waveform (Time Domain)":
            st.pyplot(plot_waveform(sig, sr=sr, title="Lecture Audio Waveform (Normalized Amplitude vs Time)"))
            st.caption("Displays instantaneous acoustic pressure variations over time. Useful for identifying dynamic range, speech pauses, and amplitude envelopes.")

        elif viz_type == "STFT Spectrogram (Frequency Domain)":
            st.pyplot(plot_spectrogram(sig, sr=sr))
            st.caption("Short-Time Fourier Transform showing spectral energy density across linear frequencies (0 - 8000 Hz).")

        elif viz_type == "Mel-Scale Spectrogram":
            st.pyplot(plot_mel_spectrogram(sig, sr=sr))
            st.caption("Mel-scaled filterbank energies mimicking human cochlear frequency sensitivity.")

        elif viz_type == "MFCC Heatmap":
            st.pyplot(plot_mfcc(sig, sr=sr))
            st.caption("Mel-Frequency Cepstral Coefficients (1-13) representing acoustic spectral shape, widely used in speech recognition and speaker diarization.")

        elif viz_type == "LPC Spectral Envelope & Formants":
            frame_sec = st.slider("Select frame center time (seconds)", 0.0, float(len(sig) / sr), float(len(sig) / (2 * sr)), step=0.1)
            order = st.slider("LPC Prediction Order (p)", 8, 20, 14)
            st.pyplot(plot_lpc_spectrum(sig, sr=sr, order=order, center_sec=frame_sec))
            st.caption("LPC All-Pole Filter Model H(z) = G / A(z) computed via Levinson-Durbin recursion. Peaks denote vocal tract formant resonances.")

        elif viz_type == "VAD & Speaker Diarization Alignment":
            dur = len(sig) / sr
            st.pyplot(plot_vad_and_diarization_timeline(dur, st.session_state.vad_intervals, st.session_state.diarization_segments))
            st.caption("Aligned visual comparison of detected speech intervals (VAD) versus attributed speaker clusters.")


# -------------------------------------------------------------
# TAB 5: SPR COURSE ALGORITHMS (VIVA)
# -------------------------------------------------------------
elif navigation == "5. SPR Course Algorithms (Viva)":
    st.header("🎓 SPR Academic Course Algorithms & Demonstrations")
    st.markdown("Interactive demonstrations of foundational Speech Processing and Recognition algorithms taught in the MSc curriculum.")

    algo_choice = st.selectbox(
        "Select Algorithm Demonstration",
        [
            "Dynamic Time Warping (DTW) - Speech Pattern Alignment",
            "Hidden Markov Model (HMM) & Viterbi Decoding",
            "Levinson-Durbin LPC Formulation",
        ],
    )

    if "Dynamic Time Warping" in algo_choice:
        st.subheader("Dynamic Time Warping (DTW)")
        st.markdown(
            "DTW computes an optimal non-linear time alignment between two speech feature sequences of different speaking rates "
            "using dynamic programming with local continuity constraints (Rabiner Unit 2 / 3)."
        )

        # Interactive DTW comparison
        c1, c2 = st.columns(2)
        with c1:
            t1 = np.linspace(0, 2 * np.pi, 30)
            seq1 = np.sin(t1) + 0.1 * np.random.randn(30)
            st.line_chart(seq1)
            st.caption("Reference Template Feature Sequence")
        with c2:
            t2 = np.linspace(0, 2 * np.pi, 45)  # Stretched / slower speaking rate
            seq2 = np.sin(t2**1.1) + 0.1 * np.random.randn(45)
            st.line_chart(seq2)
            st.caption("Test Input Feature Sequence (Warped Timing)")

        dist, path, cost_mat = dynamic_time_warping(seq1, seq2)
        st.success(f"Optimal Normalized Warping Distance: **{dist:.4f}** (Warping Path Steps: {len(path)})")

    elif "Hidden Markov Model" in algo_choice:
        st.subheader("HMM Viterbi Decoding Demo")
        st.markdown(
            "Solves Rabiner's Problem 2: Finding the single best state sequence $Q = (q_1, q_2, ..., q_T)$ for a given observation sequence "
            "$O = (o_1, o_2, ..., o_T)$ using the dynamic programming Viterbi algorithm."
        )

        states = ["Silence (S0)", "Vowel (S1)", "Consonant (S2)"]
        obs_names = ["Low Energy (O0)", "Medium Formant (O1)", "High ZCR (O2)"]

        # Synthetic observations
        obs_seq = [0, 0, 1, 1, 1, 2, 2, 0]
        st.write(f"Observation Sequence: {[obs_names[o] for o in obs_seq]}")

        # Transition and emission matrices
        pi = np.array([0.7, 0.2, 0.1])
        A = np.array([
            [0.6, 0.3, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
        ])
        B = np.array([
            [0.8, 0.1, 0.1],  # Silence emits Low Energy
            [0.1, 0.8, 0.1],  # Vowel emits Medium Formant
            [0.1, 0.2, 0.7],  # Consonant emits High ZCR
        ])

        best_path, prob, vit_mat = viterbi_decoding_demo(obs_seq, states, pi, A, B)
        st.success(f"Decoded Hidden State Sequence: **{' ➔ '.join(best_path)}**")
        st.caption(f"Sequence Likelihood Probability: {prob:.6e}")

    elif "Levinson-Durbin" in algo_choice:
        st.subheader("Linear Predictive Coding: Levinson-Durbin Formulation")
        st.latex(r"s[n] = \sum_{k=1}^{p} a_k s[n-k] + e[n]")
        st.latex(r"k_i = \frac{r(i) - \sum_{j=1}^{i-1} a_j^{(i-1)} r(i-j)}{E^{(i-1)}}")
        st.markdown(
            "The Levinson-Durbin algorithm reduces the computational complexity of solving the Toeplitz autocorrelation matrix "
            "from $\\mathcal{O}(p^3)$ via Gaussian elimination down to $\\mathcal{O}(p^2)$."
        )


# -------------------------------------------------------------
# TAB 6: EVALUATION & BENCHMARKING
# -------------------------------------------------------------
elif navigation == "6. Evaluation & Benchmarking":
    st.header("📈 Evaluation & Academic Benchmarking")
    st.markdown("Assess ASR transcription accuracy via Word Error Rate (WER) and Diarization Error Rate (DER).")

    eval_type = st.radio("Select Metric", ["Word Error Rate (WER)", "Diarization Error Rate (DER)"])

    if eval_type == "Word Error Rate (WER)":
        st.subheader("Word Error Rate (WER) Analysis")
        st.latex(r"\text{WER} = \frac{S + D + I}{N} = \frac{\text{Substitutions} + \text{Deletions} + \text{Insertions}}{\text{Reference Word Count}}")

        hyp_text = " ".join([s.text for s in st.session_state.transcript_segments])
        st.markdown("#### Generated Hypothesis Transcript")
        st.text_area("Hypothesis (ASR Output)", value=hyp_text, height=120)

        ref_text = st.text_area(
            "Enter Ground Truth Reference Transcript for Comparison:",
            placeholder="Paste ground truth transcript text here...",
            height=120,
        )

        if st.button("Compute WER"):
            if not ref_text.strip():
                st.warning("Please provide a reference transcript to calculate WER.")
            else:
                scores = calculate_wer(ref_text, hyp_text)
                if "error" in scores:
                    st.error(scores["error"])
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Word Error Rate", scores["wer_percent"])
                    c2.metric("Substitutions (S)", scores["substitutions"])
                    c3.metric("Deletions (D)", scores["deletions"])
                    c4.metric("Insertions (I)", scores["insertions"])

    elif eval_type == "Diarization Error Rate (DER)":
        st.subheader("Diarization Error Rate (DER) Analysis")
        st.latex(r"\text{DER} = \frac{T_{\text{miss}} + T_{\text{false\_alarm}} + T_{\text{speaker\_error}}}{T_{\text{total\_reference}}}")
        st.info("DER requires timestamped ground truth speaker turn annotations. Currently, predicted diarization has identified:")
        st.write(f"Total Predicted Speaker Turns: **{len(st.session_state.diarization_segments)}**")


# -------------------------------------------------------------
# TAB 7: DATA EXPORT
# -------------------------------------------------------------
elif navigation == "7. Data Export":
    st.header("💾 Export Transcripts and Metadata")
    st.markdown("Export your processed lecture transcripts into standard formats.")

    if not st.session_state.transcript_segments:
        st.warning("No transcript available to export. Upload and process a lecture recording first.")
    else:
        segs = st.session_state.transcript_segments
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            txt_data = export_to_txt(segs)
            st.download_button("📄 Download TXT", data=txt_data, file_name="lecture_transcript.txt", mime="text/plain")
        with c2:
            json_data = export_to_json(segs)
            st.download_button("📋 Download JSON", data=json_data, file_name="lecture_transcript.json", mime="application/json")
        with c3:
            csv_data = export_to_csv(segs)
            st.download_button("📊 Download CSV", data=csv_data, file_name="lecture_transcript.csv", mime="text/csv")
        with c4:
            srt_data = export_to_srt(segs)
            st.download_button("🎬 Download SRT Subtitles", data=srt_data, file_name="lecture_subtitles.srt", mime="text/plain")

        st.markdown("#### Transcript Preview")
        st.code(txt_data[:1000] + ("..." if len(txt_data) > 1000 else ""), language="markdown")


# -------------------------------------------------------------
# TAB 8: TECHNICAL DOCUMENTATION
# -------------------------------------------------------------
elif navigation == "8. Technical Documentation":
    st.header("📖 System Architecture & Technical Specifications")
    st.markdown("""
    ### System Architecture & Pipeline Workflow
    ```
    Audio Upload (WAV/MP3/FLAC) ➔ Format & Duration Validation ➔ 16kHz Resampling & Peak Normalization
             ➔ Short-Time Energy & Zero-Crossing VAD ➔ Offline Faster-Whisper ASR (CTranslate2)
             ➔ 13-MFCC + Delta Extraction ➔ Agglomerative Clustering (Cosine Distance)
             ➔ Temporal Segment Integration ➔ Synchronized HTML5 Audio Player & Search Index
    ```

    ### Key Technical Design Decisions
    1. **Local Offline Inference:** Utilizes `faster-whisper` running with 8-bit quantization (`int8`) on CPU, guaranteeing zero external API cost and no network dependencies during lecture processing.
    2. **DOM-Level Audio Synchronization:** Audio seeking and timestamp synchronization are anchored directly to the HTML5 `<audio>` element (`currentTime`, `timeupdate`), eliminating network polling drift and Streamlit re-render latencies.
    3. **Academic Explainability:** Every stage connects to the Rabiner speech processing syllabus (LPC Levinson-Durbin, MFCC filterbanks, DTW dynamic programming, and HMM Viterbi decoding).
    """)
