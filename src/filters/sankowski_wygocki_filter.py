"""SankowskiWygockiFilter -- zero-false-negative ANN structure of Sankowski &
Wygocki, "Approximate nearest neighbors search without false negatives for l2
for c > sqrt(log log n)" (ISAAC 2017, arXiv 1708.06395).

Reduction (Lemmas 3 and 7): draw `num_families` (the paper's L) independent
random orthonormal bases of R^d', d' = d zero-padded to a multiple of
`block_dim` (k; footnote 2). Family j's block i is the map

    A(i, j) = sqrt(d'/k) * (rows [i*k, (i+1)*k) of basis j) : R^d -> R^k.

Because the blocks of one basis split ||z||^2 exactly, the mean of
||A(i, j) z||^2 over i is ||z||^2, so for EVERY z and every family some block
has ||A(i, j) z|| <= ||z|| (pigeonhole) -- deterministically, not w.h.p.

Instances (Lemma 9): one table per combination (i_1, ..., i_L) in
[d'/k]^L, i.e. (d'/k)^L tables. A point's key in a table is
g(x) = (g_1(A(i_1,1) x), ..., g_L(A(i_L,L) x)), each g_j concatenating
`hashes_per_block` (w) hashes h(y) = floor(<u, y> / r) with u uniform on the
unit sphere S^(k-1) (Section 6.1, Theorem 11, the hash of Wygocki [13]),
drawn independently per table. For ||x - y|| <= r the pigeonhole combination
has every component within r, so every hash in that table differs by at most
one: the query reaches x through that table's 3^(w*L) neighbourhood.

num_families=1 is Section 5's algorithm (Theorem 6, d'/k tables);
num_families > 1 is Section 6 (Theorem 14). The paper's parameter choices
(k = O(log n) for Theorem 6, k = O(f(n) log log n) with L = log n /
(f(n) log log n) for Theorem 14) are asymptotic; here k, w and L are plain
dials. As with PacukLSHFilter, zero false negatives hold for any setting;
the dials only move FPR against memory / query time. See _floor_lsh for the
shared store/probe replication and key store.
"""

import itertools
from typing import Optional

import numpy as np

from filters._floor_lsh import FloorLSHFilter


class SankowskiWygockiFilter(FloorLSHFilter):

    def __init__(
        self,
        dim: int,
        distance_threshold: float,
        max_memory: Optional[int] = None,
        block_dim: int = 16,
        hashes_per_block: int = 8,
        num_families: int = 1,
        replicate: str = "store",
        error_rate: float = 0.01,
        slack: float = 1e-3,
        seed: int = 42,
    ):
        super().__init__(dim, distance_threshold, max_memory=max_memory, replicate=replicate,
                         error_rate=error_rate, slack=slack, seed=seed)
        if block_dim < 1 or hashes_per_block < 1 or num_families < 1:
            raise ValueError(
                f"block_dim, hashes_per_block and num_families must all be >= 1, got "
                f"{block_dim}, {hashes_per_block}, {num_families}"
            )
        self.block_dim = block_dim
        self.hashes_per_block = hashes_per_block
        self.num_families = num_families
        self.num_blocks = -(-dim // block_dim)
        self.padded_dim = self.num_blocks * block_dim

        rng = np.random.default_rng(seed)
        scale = np.sqrt(self.padded_dim / block_dim)
        # (L, d', d'): family j's rows, pre-scaled so block i of family j is
        # rows [i*k, (i+1)*k) -- A(i, j) of Lemma 3.
        bases = []
        for _ in range(num_families):
            q, r = np.linalg.qr(rng.standard_normal((self.padded_dim, self.padded_dim)))
            q *= np.sign(np.diag(r))  # Haar-uniform, not just orthonormal
            bases.append(q.T * scale)
        self._bases = np.stack(bases)

        # (T, L): block index per family, one row per table.
        self._combinations = np.array(
            list(itertools.product(range(self.num_blocks), repeat=num_families)), dtype=np.int64
        )
        # (T, L, w, k): independent unit hash directions per table and family.
        u = rng.standard_normal(
            (len(self._combinations), num_families, hashes_per_block, block_dim)
        )
        self._hash_dirs = u / np.linalg.norm(u, axis=-1, keepdims=True)

    @property
    def num_tables(self) -> int:
        return len(self._combinations)

    @property
    def signature_length(self) -> int:
        return self.num_families * self.hashes_per_block

    def _signatures(self, vectors: np.ndarray) -> np.ndarray:
        x = np.asarray(vectors, dtype=np.float64)
        if self.padded_dim > self.dim:
            x = np.pad(x, ((0, 0), (0, self.padded_dim - self.dim)))
        n, k, w = len(x), self.block_dim, self.hashes_per_block
        out = np.empty((n, self.num_tables, self.signature_length), dtype=np.int32)
        for j in range(self.num_families):
            blocks = (x @ self._bases[j].T).reshape(n, self.num_blocks, k)  # (n, d'/k, k)
            picked = blocks[:, self._combinations[:, j], :]                 # (n, T, k)
            proj = np.einsum("ntk,twk->ntw", picked, self._hash_dirs[:, j])
            out[:, :, j * w:(j + 1) * w] = np.floor(proj / self.radius)
        return out
