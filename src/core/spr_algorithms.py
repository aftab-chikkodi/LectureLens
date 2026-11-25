"""Speech Processing and Recognition (SPR) Academic Algorithms Module.
Explicitly connects to the MSc syllabus:
- Pre-emphasis
- Short-Time Fourier Transform & Filterbanks
- Linear Predictive Coding (LPC) using Levinson-Durbin recursion
- Dynamic Time Warping (DTW) pattern matching
- Discrete Hidden Markov Model (HMM) & Viterbi Decoding demonstration
- Vector Quantization (LBG / K-Means concept)
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import scipy.signal
import scipy.linalg


def compute_lpc_levinson(signal_frame: np.ndarray, order: int = 12) -> Tuple[np.ndarray, float]:
    """
    Computes Linear Predictive Coding (LPC) coefficients using the Levinson-Durbin recursion.
    Rabiner & Schafer foundation: s_hat[n] = sum_{k=1}^p a_k * s[n-k]
    """
    if len(signal_frame) < order:
        return np.zeros(order + 1), 0.0

    # Auto-correlation of windowed frame
    window = np.hamming(len(signal_frame))
    frame = signal_frame * window
    r = np.correlate(frame, frame, mode="full")
    r = r[len(frame) - 1 :]  # Non-negative lags 0..len-1

    r_order = r[: order + 1]
    if r_order[0] <= 1e-10:
        return np.zeros(order + 1), 0.0

    # Levinson-Durbin recursion
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r_order[0]

    for i in range(1, order + 1):
        # Calculate reflection coefficient k_i
        acc = sum(a[j] * r_order[i - j] for j in range(1, i))
        k = (r_order[i] - acc) / e
        a[i] = k
        for j in range(1, (i // 2) + 1):
            temp1 = a[j] - k * a[i - j]
            temp2 = a[i - j] - k * a[j]
            a[j] = temp1
            a[i - j] = temp2
        e = e * (1.0 - k * k)
        if e <= 0:
            break

    # Convention: in LPC inverse filter A(z) = 1 - sum(a_k z^-k), so lpc_coeffs = [1, -a1, -a2, ...]
    lpc_coeffs = np.zeros(order + 1)
    lpc_coeffs[0] = 1.0
    lpc_coeffs[1:] = -a[1:]

    return lpc_coeffs, float(e)


def dynamic_time_warping(
    seq_a: np.ndarray, seq_b: np.ndarray, metric: str = "euclidean"
) -> Tuple[float, List[Tuple[int, int]], np.ndarray]:
    """
    Genuine Dynamic Time Warping (DTW) for time alignment of speech feature vectors.
    Solves optimal alignment path between templates of differing duration.
    Rabiner SPR Unit 2 / 3 concept.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return 0.0, [], np.zeros((1, 1))

    # Pairwise cost matrix
    if seq_a.ndim == 1:
        cost_mat = np.abs(seq_a[:, None] - seq_b[None, :])
    else:
        # Vector euclidean distance
        diff = seq_a[:, None, :] - seq_b[None, :, :]
        cost_mat = np.sqrt(np.sum(diff**2, axis=-1))

    # Accumulated cost matrix
    accum_cost = np.full((n, m), np.inf)
    accum_cost[0, 0] = cost_mat[0, 0]

    for i in range(1, n):
        accum_cost[i, 0] = accum_cost[i - 1, 0] + cost_mat[i, 0]
    for j in range(1, m):
        accum_cost[0, j] = accum_cost[0, j - 1] + cost_mat[0, j]

    for i in range(1, n):
        for j in range(1, m):
            accum_cost[i, j] = cost_mat[i, j] + min(
                accum_cost[i - 1, j],      # Insertion
                accum_cost[i, j - 1],      # Deletion
                accum_cost[i - 1, j - 1],  # Match
            )

    # Backtracking optimal warping path
    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [
                (accum_cost[i - 1, j - 1], (i - 1, j - 1)),
                (accum_cost[i - 1, j], (i - 1, j)),
                (accum_cost[i, j - 1], (i, j - 1)),
            ]
            _, (i, j) = min(candidates, key=lambda x: x[0])
        path.append((i, j))

    path.reverse()
    normalized_distance = accum_cost[n - 1, m - 1] / (n + m)
    return float(normalized_distance), path, accum_cost


def viterbi_decoding_demo(
    observations: List[int],
    states: List[str],
    initial_prob: np.ndarray,
    trans_matrix: np.ndarray,
    emission_matrix: np.ndarray,
) -> Tuple[List[str], float, np.ndarray]:
    """
    Genuine Viterbi Algorithm implementation for discrete HMM decoding.
    Finds the most likely sequence of hidden states (e.g. phoneme/word states) given acoustic observations.
    Demonstrates Rabiner's three fundamental HMM problems (specifically Problem 2: Decoding).
    """
    n_states = len(states)
    n_obs = len(observations)
    if n_obs == 0:
        return [], 0.0, np.zeros((1, 1))

    # Log space to prevent underflow
    eps = 1e-12
    viterbi_mat = np.zeros((n_states, n_obs))
    backpointer = np.zeros((n_states, n_obs), dtype=int)

    # Initialization
    viterbi_mat[:, 0] = np.log(initial_prob + eps) + np.log(emission_matrix[:, observations[0]] + eps)

    # Recursion
    for t in range(1, n_obs):
        obs = observations[t]
        for s in range(n_states):
            # trans_matrix[:, s] is transition from all previous states to current s
            trans_probs = viterbi_mat[:, t - 1] + np.log(trans_matrix[:, s] + eps)
            best_prev = np.argmax(trans_probs)
            viterbi_mat[s, t] = trans_probs[best_prev] + np.log(emission_matrix[s, obs] + eps)
            backpointer[s, t] = best_prev

    # Termination
    best_last_state = np.argmax(viterbi_mat[:, -1])
    best_log_prob = viterbi_mat[best_last_state, -1]

    # Traceback
    best_path_indices = [best_last_state]
    for t in range(n_obs - 1, 0, -1):
        best_path_indices.append(backpointer[best_path_indices[-1], t])
    best_path_indices.reverse()

    best_path = [states[idx] for idx in best_path_indices]
    return best_path, float(np.exp(best_log_prob)), viterbi_mat
