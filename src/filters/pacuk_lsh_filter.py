"""PacukLSHFilter -- zero-false-negative LSH of Pacuk, Sankowski, Wegrzycki &
Wygocki, "Locality-Sensitive Hashing without False Negatives for l_p"
(COCOON 2016, arXiv 1611.09317), instantiated for l2.

Hash (Section 3): h(x) = floor(<x, v> / (r * rho)), rho = d^(1 - 1/p) =
sqrt(d) for l2, v with iid coordinates on {-1, +1} (Lemma 2, the variant the
paper's own discussion uses) or U(-1, 1) (Lemma 1, distribution="uniform").
Since |v_i| <= 1,

    |<x - y, v>| <= ||x - y||_1 <= sqrt(d) * ||x - y||_2,

so ||x - y||_2 <= r puts x and y in the same or adjacent buckets
(Observation 3), with certainty. g(x) concatenates `num_hashes` (the paper's
k) such hashes into ONE table (Theorem 1) -- no OR over tables is needed,
because nothing is left to chance on the close side.

The paper's c-approximation bound needs c > sqrt(d)-order (c > 2*sqrt(d)
with Wygocki's refinement), which on unit-norm data at small tau exceeds the
sphere's diameter -- so its FPR guarantee is vacuous in this project's regime.
The no-false-negative property does not depend on c or on num_hashes: those
only move FPR against memory / query time. See _floor_lsh for the shared
store/probe replication and key store.
"""

from typing import Optional

import numpy as np

from filters._floor_lsh import FloorLSHFilter


_DISTRIBUTIONS = ("rademacher", "uniform")


class PacukLSHFilter(FloorLSHFilter):

    def __init__(
        self,
        dim: int,
        distance_threshold: float,
        max_memory: Optional[int] = None,
        num_hashes: int = 8,
        distribution: str = "rademacher",
        replicate: str = "store",
        error_rate: float = 0.01,
        slack: float = 1e-3,
        seed: int = 42,
    ):
        super().__init__(dim, distance_threshold, max_memory=max_memory, replicate=replicate,
                         error_rate=error_rate, slack=slack, seed=seed)
        if num_hashes < 1:
            raise ValueError(f"num_hashes must be >= 1, got {num_hashes}")
        if distribution not in _DISTRIBUTIONS:
            raise ValueError(f"distribution must be one of {_DISTRIBUTIONS}, got {distribution!r}")
        self.num_hashes = num_hashes
        self.distribution = distribution

        rng = np.random.default_rng(seed)
        if distribution == "rademacher":
            v = rng.choice(np.array([-1.0, 1.0]), size=(num_hashes, dim))
        else:
            v = rng.uniform(-1.0, 1.0, size=(num_hashes, dim))
        self._projections = v  # float64: see _signatures
        self.bucket_width = self.radius * np.sqrt(dim)  # r * rho, rho = d^(1 - 1/2)

    @property
    def num_tables(self) -> int:
        return 1

    @property
    def signature_length(self) -> int:
        return self.num_hashes

    def _signatures(self, vectors: np.ndarray) -> np.ndarray:
        # float64 so the projection's own rounding stays far below `slack`.
        proj = np.asarray(vectors, dtype=np.float64) @ self._projections.T
        return np.floor(proj / self.bucket_width).astype(np.int32)[:, None, :]
