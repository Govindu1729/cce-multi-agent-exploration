"""Protocol analysis: message frequency, entropy, and mutual information."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_selection import mutual_info_classif


def message_statistics(
    message_indices: np.ndarray,
    send_flags: np.ndarray,
    codebook_size: int,
) -> dict[str, float]:
    """Compute basic message statistics.

    Args:
        message_indices: (T, B, N, S) int — VQ codebook indices per agent per timestep.
        send_flags: (T, B, N) bool — which agents sent messages.
        codebook_size: K, number of codes in the codebook.

    Returns:
        dict with frequency, entropy, diversity metrics.
    """
    # Flatten
    flat_idx = message_indices.reshape(-1)
    flat_send = send_flags.reshape(-1)

    # Only analyse messages that were actually sent
    sent_idx = flat_idx[flat_send]

    total_possible = flat_idx.size
    total_sent = sent_idx.size

    # Frequency
    send_rate = total_sent / total_possible if total_possible > 0 else 0.0

    if len(sent_idx) == 0:
        return {
            "send_rate": 0.0,
            "entropy": 0.0,
            "active_codes": 0,
            "code_diversity": 0.0,
        }

    # Histogram of code usage
    counts = np.bincount(sent_idx, minlength=codebook_size)
    probs = counts / counts.sum()

    # Entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    # Active codes (used at least once)
    active_codes = (counts > 0).sum()

    # Diversity = active_codes / codebook_size
    diversity = active_codes / codebook_size

    return {
        "send_rate": float(send_rate),
        "entropy": float(entropy),
        "active_codes": int(active_codes),
        "code_diversity": float(diversity),
        "total_messages": int(total_sent),
    }


def message_mutual_information(
    message_indices: np.ndarray,
    state_features: np.ndarray,
    send_flags: np.ndarray,
) -> dict[str, float]:
    """Compute mutual information between messages and state features.

    Args:
        message_indices: (T, B, N, S) int
        state_features: (T, B, N, F) float — state features (e.g., coverage, distance to obstacles)
        send_flags: (T, B, N) bool

    Returns:
        dict with MI values for each feature.
    """
    # Flatten to (T*B*N, S) and (T*B*N, F)
    flat_msg = message_indices.reshape(-1, message_indices.shape[-1])
    flat_state = state_features.reshape(-1, state_features.shape[-1])
    flat_send = send_flags.reshape(-1)

    # Only use samples where message was sent
    sent_msg = flat_msg[flat_send]
    sent_state = flat_state[flat_send]

    if len(sent_msg) < 10:
        return {"mi_mean": 0.0, "n_samples": 0}

    # For each code slot, compute MI with each state feature
    # Average over slots
    n_slots = sent_msg.shape[1]
    n_features = sent_state.shape[1]
    mi_per_slot = np.zeros(n_slots)

    for s in range(n_slots):
        try:
            # MI between discrete message and continuous feature
            mi = mutual_info_classif(
                sent_state, sent_msg[:, s], discrete_features=False, n_neighbors=3
            )
            mi_per_slot[s] = mi.mean()
        except Exception:
            mi_per_slot[s] = 0.0

    return {
        "mi_mean": float(mi_per_slot.mean()),
        "mi_per_slot": mi_per_slot.tolist(),
        "n_samples": len(sent_msg),
    }


def message_entropy_over_time(
    message_indices: np.ndarray,
    send_flags: np.ndarray,
    codebook_size: int,
    window_size: int = 10,
) -> np.ndarray:
    """Compute message entropy over time using a sliding window.

    Args:
        message_indices: (T, B, N, S) int
        send_flags: (T, B, N) bool
        codebook_size: K
        window_size: number of timesteps per window.

    Returns:
        (n_windows,) array of entropy values.
    """
    T = message_indices.shape[0]
    n_windows = max(1, T // window_size)
    entropies = []

    for w in range(n_windows):
        start = w * window_size
        end = min(start + window_size, T)
        window_idx = message_indices[start:end].reshape(-1)
        window_send = send_flags[start:end].reshape(-1)
        sent = window_idx[window_send]

        if len(sent) == 0:
            entropies.append(0.0)
            continue

        counts = np.bincount(sent, minlength=codebook_size)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        entropies.append(float(entropy))

    return np.array(entropies)
