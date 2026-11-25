# LectureLens: Comprehensive Viva Examination & Defense Guide

**MSc Artificial Intelligence and Machine Learning**  
**Course:** Speech Processing and Recognition (SPR)  

---

### Q1: Why did you convert audio to 16 kHz mono during preprocessing?
**Answer:**
Human speech conveys almost all phonetic and linguistic information below 8 kHz (Nyquist criterion: $f_s \ge 2 f_{\max}$, hence $16\text{ kHz}$ captures up to $8\text{ kHz}$). Standard acoustic models (including Whisper, Kaldi, and Wav2Vec) are pre-trained on 16 kHz single-channel audio. Converting stereo to mono by channel averaging eliminates phase interference while cutting matrix processing sizes in half.

---

### Q2: How does the Voice Activity Detection (VAD) algorithm distinguish speech from noise?
**Answer:**
We implement a dual-criterion algorithm based on Rabiner & Schafer:
1. **Short-Time Energy (STE):**
   $$E_n = \sum_{m} [x[m] w[n-m]]^2$$
   Speech voiced segments exhibit substantially higher energy than ambient background noise.
2. **Zero-Crossing Rate (ZCR):**
   $$Z_n = \frac{1}{2N} \sum_{m=1}^{N-1} |\text{sgn}(x[m]) - \text{sgn}(x[m-1])|$$
   Unvoiced consonants (e.g., /s/, /t/, /sh/) exhibit lower energy but a high zero-crossing rate.
3. **Hangover Time:** Short pauses between syllables or words (typically $<300\text{ ms}$) are bridged using hangover smoothing so that natural speech flow is not fragmented.

---

### Q3: What is the purpose of Pre-Emphasis in speech processing?
**Answer:**
The glottal airflow excitation signal exhibits an approximate $-6\text{ dB/octave}$ roll-off due to glottal volume velocity. Furthermore, lip radiation introduces a $+6\text{ dB/octave}$ effect. Pre-emphasis applies a first-order high-pass filter:
$$y[n] = x[n] - a \cdot x[n-1], \quad 0.95 \le a \le 0.98$$
This boosts high-frequency formants, flattens the overall spectral tilt, and improves numerical stability for subsequent LPC and MFCC computation.

---

### Q4: Explain the mathematical significance of the Levinson-Durbin recursion in LPC.
**Answer:**
Linear Predictive Coding models the vocal tract as an all-pole digital filter $H(z) = G / A(z)$. Minimizing mean squared prediction error yields the Yule-Walker equations:
$$\sum_{k=1}^p a_k r(|i-k|) = r(i), \quad i = 1, \dots, p$$
The autocorrelation matrix $R$ is Toeplitz (symmetric and constant along diagonals). Standard Gaussian elimination requires $\mathcal{O}(p^3)$ operations. The **Levinson-Durbin recursion** leverages this Toeplitz symmetry to solve for prediction coefficients $\{a_k\}$ and reflection coefficients $\{k_i\}$ recursively in $\mathcal{O}(p^2)$ complexity.

---

### Q5: How does Dynamic Time Warping (DTW) handle speech rate variations?
**Answer:**
Two utterances of the same word naturally differ in total duration and local syllable stretching. DTW constructs a pairwise acoustic distance matrix $C \in \mathbb{R}^{N \times M}$ and uses dynamic programming to solve the recurrence:
$$D(i, j) = c(i, j) + \min \{ D(i-1, j), D(i, j-1), D(i-1, j-1) \}$$
subject to:
1. **Boundary constraints:** $D(0, 0)$ to $D(N-1, M-1)$.
2. **Monotonicity:** $i_k \ge i_{k-1}$ and $j_k \ge j_{k-1}$.
3. **Local continuity:** Warping steps are restricted to horizontal, vertical, or diagonal increments.

---

### Q6: How does the Viterbi Algorithm work in Hidden Markov Models?
**Answer:**
For an HMM parameter set $\lambda = (A, B, \pi)$, Viterbi solves Problem 2 (decoding the optimal hidden state sequence $Q^*$ that maximizes $P(Q | O, \lambda)$).
It defines the induction variable:
$$\delta_t(j) = \max_{q_1, \dots, q_{t-1}} P(q_1 \dots q_t = S_j, o_1 \dots o_t | \lambda)$$
Induction step:
$$\delta_{t}(j) = \left[ \max_{1 \le i \le N} (\delta_{t-1}(i) a_{ij}) \right] b_j(o_t)$$
Backpointers $\psi_t(j)$ record the maximizing predecessor state at each step, enabling backtracking of the optimal state sequence in linear time $\mathcal{O}(N^2 T)$ rather than exponential search $\mathcal{O}(N^T)$.

---

### Q7: How is client-side audio synchronization achieved without lag?
**Answer:**
Streamlit runs on a Python server where rerendering widgets introduces noticeable network roundtrip latency. To achieve true instant synchronization:
1. We embedded a custom HTML5/JavaScript player component into the DOM.
2. The browser's native `<audio>` element serves as the single source of truth.
3. Event listeners on `timeupdate` compare `audio.currentTime` against segment intervals $[start, end]$ and highlight the active segment smoothly using `scrollIntoView({ behavior: 'smooth', block: 'nearest' })`.
4. Clicking any transcript card immediately updates `audio.currentTime = start` on the client side with zero server overhead.

---

### Q8: How is Speaker Diarization performed?
**Answer:**
1. **Feature Extraction:** For each speech segment identified by VAD, we compute 13 MFCCs plus their delta coefficients ($\Delta$) and calculate temporal means and variances to form a 26-dimensional speaker fingerprint.
2. **Clustering:** We apply Agglomerative Hierarchical Clustering with cosine distance metric and average linkage.
3. **Model Selection:** When the speaker count is unknown, the silhouette score over cluster counts $k \in [2, 5]$ dynamically identifies the optimal cluster separation.
4. **Integration:** Transcript word/segment boundaries are mapped to diarization intervals via maximum temporal overlap.
