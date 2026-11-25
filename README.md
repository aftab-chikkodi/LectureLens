# LectureLens: Intelligent Lecture Navigation & Speaker-Aware Search

**MSc Artificial Intelligence and Machine Learning**  
**Course:** Speech Processing and Recognition (SPR)  

---

## 1. Overview & Problem Statement

University lectures, seminars, and technical conferences are predominantly delivered as lengthy, continuous audio-video recordings. Navigating these recordings for exam preparation, revision, or topic extraction is notoriously inefficient. Students routinely waste time manually scrubbing through timelines to locate specific definitions, speaker questions, or equations.

**LectureLens** provides an automated, locally-executable solution that transforms raw multi-speaker lecture recordings into an **interactive, speaker-aware, searchable, and timestamp-synchronized learning interface**.

---

## 2. System Architecture & Complete Pipeline

```
┌─────────────────────────┐
│ Lecture Audio Upload    │ (.wav, .mp3, .flac, .m4a)
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Integrity Validation    │ Format, Duration, Sample Rate & Dynamic Range Checks
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Acoustic Preprocessing  │ 16 kHz Resampling, Mono-conversion & Peak Normalization
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Voice Activity Detection│ Short-Time Energy (STE) & Zero-Crossing Rate (ZCR) with Hangover
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Automatic Speech Recog. │ Local Faster-Whisper (CTranslate2 int8 CPU Quantization)
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Speaker Diarization     │ 13-MFCC + Deltas & Agglomerative Cosine Clustering
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Temporal Integration    │ Interval overlap merging (Text + Timestamps + Speaker IDs)
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Synchronized Audio & UI │ DOM-level HTML5 Audio Player + Interactive Transcript + Search
└─────────────────────────┘
```

---

## 3. SPR Academic Course Connections

The implementation bridges classical speech signal processing concepts directly with modern deep neural acoustic modeling:

1. **Speech Signal Representation & Pre-Emphasis:** High-frequency boost filter ($y[n] = x[n] - 0.97 x[n-1]$) compensating for glottal volume velocity roll-off.
2. **Short-Time Analysis:** Hamming windowing ($25\text{ ms}$ frames with $10\text{ ms}$ hop step) for quasi-stationary speech assumptions.
3. **Voice Activity Detection (VAD):** Dual-parameter thresholding using Short-Time Energy and Zero-Crossing Rates (Rabiner & Schafer).
4. **Spectral & Cepstral Features:** Mel-scaled triangular filterbank energies and Discrete Cosine Transform for Mel-Frequency Cepstral Coefficients (MFCC 1-13 + $\Delta$).
5. **Linear Predictive Coding (LPC):** Vocal tract all-pole filter modeling ($H(z) = G / A(z)$) solved in $\mathcal{O}(p^2)$ complexity via the Levinson-Durbin recursion algorithm.
6. **Dynamic Time Warping (DTW):** Optimal dynamic programming alignment of temporal speech vectors under local path constraints.
7. **Hidden Markov Models (HMM):** Complete implementation of Rabiner's Problem 2 (Viterbi decoding) finding $\arg\max_Q P(Q | O, \lambda)$.
8. **ASR Evaluation:** Word Error Rate (WER) with Levenshtein alignment ($S, D, I$) and Diarization Error Rate (DER).

---

## 4. Installation & Execution Guide

### Prerequisites
- Python 3.11 (configured in `.venv`)
- `uv` package manager (optional, standard pip is supported)

### Quick Start
```bash
# 1. Activate isolated Python 3.11 virtual environment
.venv\Scripts\activate

# 2. Launch the Streamlit application
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 5. Running Automated Tests

To run the unit test suite:
```bash
.venv\Scripts\pytest tests/test_core.py -v
```

To run the end-to-end integration test:
```bash
.venv\Scripts\python tests/test_e2e_pipeline.py
```
