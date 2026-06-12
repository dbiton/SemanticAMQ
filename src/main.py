import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sys

# Your specific imports
from filters.pyfusefilter_filters import Xor8Filter, Xor16Filter, Fuse8Filter, Fuse16Filter
from filters.cuckoo_filter import CuckooFilter
from filters.lsh_bloom import LSHBloom
from filters.bloom_filter import BloomFilter
from filters.naive_filter import NaiveFilter
from lsh import CrossPolytopeLSH, SimHashLSH
from util.fetch_datasets import load_embeds


def calculate_distance_threshold(vectors: np.ndarray, percentage: float) -> float:
    """
    Calculate the distance threshold such that `percentage`% of all vector
    pairs are considered duplicates (i.e., closer than the threshold).

    Strategy: sample a representative subset of pairwise L2 distances and
    return the `percentage`-th percentile of that distribution.

    Args:
        vectors:    Array of shape (N, D) containing the embedding vectors.
        percentage: Target duplicate rate in [0, 100].

    Returns:
        A float distance threshold.
    """
    n = len(vectors)

    # Sampling cap: computing all N*(N-1)/2 pairs is O(N^2).
    # For large N we sample a random subset that is still statistically robust.
    max_sample = 5000
    if n <= max_sample:
        sample = vectors
    else:
        idx = np.random.choice(n, size=max_sample, replace=False)
        sample = vectors[idx]

    # Compute pairwise squared-L2 distances via the identity
    #   ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    sq_norms = np.sum(sample ** 2, axis=1)          # (M,)
    dot      = sample @ sample.T                     # (M, M)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * dot

    # Numerical noise can push tiny values negative
    np.clip(sq_dists, 0, None, out=sq_dists)

    # Extract upper triangle (excluding the diagonal) to avoid self-pairs
    upper_idx  = np.triu_indices(len(sample), k=1)
    distances  = np.sqrt(sq_dists[upper_idx])        # L2 distances

    threshold = float(np.percentile(distances, percentage))
    return threshold


def compute_actual_results(vectors: np.ndarray, distance_threshold: float) -> list[bool]:
    """
    Compute ground-truth duplicate labels using the same online logic that
    the filter uses:

        - The first vector is never a duplicate  → False
        - For each subsequent vector v, it is a duplicate if its L2 distance
          to *any* previously accepted (non-duplicate) vector is ≤ threshold.
          If it is NOT a duplicate it is added to the reference set.

    This mirrors the filter loop:
        predicted_results = [False]
        for v in vectors[1:]:
            is_dup = filter_instance.query(v)
            predicted_results.append(is_dup)
            if not is_dup:
                filter_instance.insert(v)

    Args:
        vectors:            Array of shape (N, D).
        distance_threshold: Maximum L2 distance to be considered a duplicate.

    Returns:
        List of N booleans, one per vector.
    """
    actual_results = [False]          # first vector is never a duplicate
    reference_set  = [vectors[0]]     # accepted (unique) vectors so far

    for v in vectors[1:]:
        ref_array = np.array(reference_set)           # (R, D)
        diffs     = ref_array - v                     # (R, D)
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))  # (R,)

        is_dup = bool(np.any(distances <= distance_threshold))
        actual_results.append(is_dup)

        if not is_dup:
            reference_set.append(v)

    return actual_results


def main():
    filters = [Fuse16Filter, LSHBloom]
    lsh_families = [CrossPolytopeLSH]

    num_vectors = 100
    dim = 384
    min_percentage_duplicates = 10
    max_percentage_duplicates = 90
    count_steps = 10
    bits = 64

    dataset_name = 'StackOverflow'
    dataset = load_embeds(dataset_name, num_vectors)
    vectors = dataset[0]

    results_list = []
    for percentage in np.linspace(min_percentage_duplicates, max_percentage_duplicates, count_steps):
        distance_threshold = calculate_distance_threshold(vectors, percentage)

        # ── Ground-truth labels for this threshold ──────────────────────────
        actual_results = compute_actual_results(vectors, distance_threshold)
        # ────────────────────────────────────────────────────────────────────

        for filter_cls in filters:
            for lsh_cls in lsh_families:
                config_label = f"{filter_cls.__name__} + {lsh_cls.__name__}"
                print(f"Evaluating {config_label} @ {percentage:.1f}% ...")

                # Initialize LSH with varying size
                if lsh_cls == SimHashLSH:
                    lsh_instance = lsh_cls(dim=dim, num_bits=bits)
                else:
                    lsh_instance = lsh_cls(dim=dim, num_rotations=max(1, bits // 16))

                filter_instance = filter_cls(dim=dim, lsh=lsh_instance)
                filter_instance.construct(np.array([vectors[0]]))  # Insert the first vector as the initial reference
                predicted_results = [False]

                for v in vectors[1:]:
                    is_dup = filter_instance.query(np.array([v]))
                    predicted_results.append(is_dup)
                    if not is_dup:
                        filter_instance.insert(np.array([v]))

                tp = sum(1 for p, a in zip(predicted_results, actual_results) if     p and     a)
                fp = sum(1 for p, a in zip(predicted_results, actual_results) if     p and not a)
                fn = sum(1 for p, a in zip(predicted_results, actual_results) if not p and     a)
                tn = sum(1 for p, a in zip(predicted_results, actual_results) if not p and not a)

                fpr       = fp / (fp + tn)                    if (fp + tn) > 0 else 0
                fnr       = fn / (fn + tp)                    if (fn + tp) > 0 else 0
                precision = tp / (tp + fp)                    if (tp + fp) > 0 else 0
                recall    = tp / (tp + fn)                    if (tp + fn) > 0 else 0
                f1_score  = (2 * precision * recall
                             / (precision + recall))          if (precision + recall) > 0 else 0

                results_list.append({
                    'Config':     config_label,
                    'Percentage': percentage,
                    'Threshold':  distance_threshold,
                    'Bits':       bits,
                    'FPR':        fpr,
                    'FNR':        fnr,
                    'Precision':  precision,
                    'Recall':     recall,
                    'F1-Score':   f1_score,
                })

    print("Evaluation completed.")
    return results_list


if __name__ == "__main__":
    main()