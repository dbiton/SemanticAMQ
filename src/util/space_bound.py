"""The information-theoretic space lower bound of the paper's Theorem
(Information-Theoretic Space Lower Bound), evaluated numerically.

Any structure that encodes k-point subsets of S^{d-1} into B bits with zero false
negatives and false-positive mass at most eps satisfies

    B >= log2 C(M, k) - log2 C(L, k) >= k log2(M / L)    (when M >= L >= k),
    M = ceil(1 / sigma(2 alpha)),   L = floor((eps + k sigma(alpha)) / sigma(alpha)),

sigma the normalized cap measure and alpha = 2 arcsin(tau / 2). The bound is informative
only if M > L, i.e. eps < sigma(alpha) / sigma(2 alpha) - k sigma(alpha): at d = 384 that
is about 1e-110 for cosine distance 0.02, far below the accept-region measure any filter at
these budgets reaches (about 1e-40). `space_lower_bound` reports that, in log space.

    PYTHONPATH=src python src/util/space_bound.py [results/budget.csv] [out.csv]
"""
import math
import sys
from typing import Dict

import numpy as np

from filters.cap_cover.cap_geometry import log10_cap_measure

_LOG2_10 = math.log2(10.0)


def alpha_of_tau(tau: float) -> float:
    """The angle a Euclidean distance tau subtends on the unit sphere."""
    return 2.0 * math.asin(min(1.0, tau / 2.0))


def space_lower_bound(dim: int, tau: float, k: int, eps: float) -> Dict[str, float]:
    """The bound of the theorem at k points, false-positive mass eps (a share of the
    uniform measure on the sphere) and threshold tau, in bits, with its ingredients.

    Returns a dict: log2_M, log2_L, bits (the bound, 0 if vacuous), vacuous (True if
    M <= L), log10_eps_max (log10 of the largest eps for which the bound is positive)
    and log10_sigma_alpha. Everything is computed in log space; sigma(alpha) is ~1e-270 at
    d = 384."""
    alpha = alpha_of_tau(tau)
    log10_sa = float(log10_cap_measure(alpha, dim)[0])
    log10_s2a = float(log10_cap_measure(min(2.0 * alpha, math.pi), dim)[0])
    log2_M = -log10_s2a * _LOG2_10                      # ceil(1/sigma(2 alpha)) ~ 1/sigma(2 alpha)
    # log10 of (eps + k sigma(alpha)) / sigma(alpha) = log10(eps / sigma(alpha) + k)
    log10_ratio = float(np.logaddexp(math.log(eps) - log10_sa * math.log(10.0), math.log(k)) / math.log(10.0))
    log2_L = log10_ratio * _LOG2_10
    vacuous = log2_M <= log2_L
    # M > L  <=>  eps < sigma(alpha) / sigma(2 alpha) - k sigma(alpha)
    log10_eps_max = log10_sa - log10_s2a
    return {"log2_M": log2_M, "log2_L": log2_L, "bits": 0.0 if vacuous else k * (log2_M - log2_L),
            "vacuous": vacuous, "log10_eps_max": log10_eps_max, "log10_sigma_alpha": log10_sa}


def measured_points(budget_csv: str) -> "pd.DataFrame":
    """The bound at each measured (corpus, budget, tau) of results/budget.csv: k = the
    reference set, eps = the filter's accept-region measure (10 ** log10_j_tau, an upper
    bound on the uniform measure it accepts), against the filter's own bits."""
    import pandas as pd
    d = pd.read_csv(budget_csv)
    rows = []
    for (dataset, kf, tau), g in d.groupby(["dataset", "k_fraction", "tau"]):
        n = int(g.n_reference.iloc[0])
        eps = 10.0 ** float(g.log10_j_tau.mean())
        b = space_lower_bound(384, float(tau), n, eps)
        rows.append({"dataset": dataset, "k_fraction": kf, "tau": tau, "n": n,
                     "log10_eps": float(g.log10_j_tau.mean()), "filter_bits": 8.0 * float(g.memory_bytes.mean()),
                     **b})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "results/budget.csv"
    table = measured_points(src)
    if len(sys.argv) > 2:
        table.to_csv(sys.argv[2], index=False)
    print(table.to_string())
