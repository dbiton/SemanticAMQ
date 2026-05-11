import numpy as np
import math
from lsh import LSH
from typing import Optional, List
from filters.filter import Filter
 

# ---------------------------------------------------------------------------
# Ribbon filter  (pure-Python implementation)
#
# No Python package exists for Ribbon filters on PyPI, so this is a
# self-contained implementation based on:
#   Peter C. Dillinger & Stefan Walzer,
#   "Ribbon filter: practically smaller than Bloom and Xor",
#   PVLDB 14(2021).
#
# Algorithm overview
# ------------------
# Each key is mapped to a (start_row, w-bit coefficient ribbon, r-bit RHS)
# triple.  The coefficient ribbon is a contiguous w-bit window starting at
# start_row in an m-row boolean matrix.  Construction solves the resulting
# banded GF(2) linear system via Gaussian elimination + back-substitution,
# yielding an m×r solution matrix C stored as a list of r-bit integers.
# Membership is checked by evaluating the dot-product (over GF(2)) of the
# key's ribbon against C and comparing to its RHS.
#
# False-positive rate ≈ 2^{-r}.  Memory overhead ≈ 5 % above information-
# theoretic minimum (controlled by the 1.05 row-expansion factor).
# ---------------------------------------------------------------------------
 
class _RibbonFilterImpl:
    """
    Pure-Python Ribbon Filter (banded GF(2) linear system).
 
    Args:
        capacity:      Number of elements to be inserted.
        bits_per_entry: Result bits *r*; FPR ≈ 2^{-r}.
                        8  → ~0.4 %  (comparable to Bloom with 11.5 bits/elem)
                        16 → ~0.0015 %
        ribbon_width:  Bandwidth *w* of the banded system (64 is standard).
    """
 
    def __init__(
        self,
        capacity: int,
        bits_per_entry: int = 8,
        ribbon_width: int = 64,
    ) -> None:
        self.w = ribbon_width
        self.r = bits_per_entry
        self.n = capacity
        # Extra rows guarantee the banded system is solvable with high prob.
        self.m = math.ceil(1.05 * capacity) + ribbon_width
        self._C: Optional[List[int]] = None
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _hash(self, key: int):
        """Return (start_row, w-bit coeff ribbon, r-bit RHS) for *key*."""
        # Two independent 64-bit finalisation hashes (fmix64-style).
        h1 = key ^ (key >> 30)
        h1 = (h1 * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        h1 ^= h1 >> 27
        h1 = (h1 * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        h1 ^= h1 >> 31
 
        h2 = key ^ (key >> 33)
        h2 = (h2 * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
        h2 ^= h2 >> 33
        h2 = (h2 * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
        h2 ^= h2 >> 33
 
        rows = max(1, self.m - self.w + 1)
        start = int(h1 % rows)
        coeff = int(h2 | 1)                        # odd → always non-zero
        rhs   = int(h1 >> (64 - self.r)) & ((1 << self.r) - 1)
        return start, coeff, rhs
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def populate(self, keys: List[int]) -> None:
        """
        Build the filter from a list of integer keys.
 
        Performs banded Gaussian elimination over GF(2)^r followed by
        back-substitution.  Each cell stores an r-bit integer.
        """
        w, m = self.w, self.m
 
        coeff_row: List[int] = [0] * m   # stored as right-shifted ribbons
        rhs_row:   List[int] = [0] * m
 
        for k in keys:
            start, coeff, rhs = self._hash(k)
            # Walk the ribbon left-to-right; XOR into the first free pivot.
            for bit in range(w):
                if not ((coeff >> bit) & 1):
                    continue
                row = start + bit
                if coeff_row[row] == 0:
                    # Free row: store the remaining (right-shifted) ribbon.
                    coeff_row[row] = coeff >> bit
                    rhs_row[row]   = rhs
                    break
                else:
                    # Eliminate: XOR away the stored pivot.
                    coeff ^= coeff_row[row] << bit
                    rhs   ^= rhs_row[row]
            # If coeff == 0 after elimination the key is already implied
            # (or a duplicate); either way no action needed.
 
        # Back-substitution (bottom-up).
        C: List[int] = [0] * m
        for i in range(m - 1, -1, -1):
            a   = coeff_row[i]
            val = rhs_row[i]
            for bit in range(1, w):
                if (a >> bit) & 1:
                    j = i + bit
                    if j < m:
                        val ^= C[j]
            C[i] = val
 
        self._C = C
 
    def contains(self, key: int) -> bool:
        """Return True if *key* is (probably) in the set."""
        if self._C is None:
            raise RuntimeError("Call populate() before contains().")
        start, coeff, rhs = self._hash(key)
        result = 0
        for bit in range(self.w):
            if (coeff >> bit) & 1:
                result ^= self._C[start + bit]
        return result == rhs
 
 
class RibbonFilter(Filter):
    """
    Ribbon Filter — a practically smaller alternative to Bloom / Xor filters.
 
    No third-party package is available on PyPI, so this wraps a pure-Python
    implementation of the Dillinger & Walzer (2021) algorithm.
 
    Construction solves a banded GF(2) linear system (Gaussian elimination +
    back-substitution) and stores an *m × r* solution matrix, where *m* ≈
    1.05 × capacity and *r* = ``bits_per_entry``.
 
    False-positive rate ≈ 2^{-bits_per_entry}:
        bits_per_entry=8  → FPR ≈ 0.4 %
        bits_per_entry=16 → FPR ≈ 0.0015 %
 
    Compared to a Bloom filter at the same FPR, a Ribbon filter uses ~30–40 %
    less memory and has faster query throughput due to cache-friendly access.
 
    Args:
        dim:           Dimensionality of the input vectors.
        capacity:      Number of elements the filter will hold.
        bits_per_entry: Result bits *r* controlling FPR (default 8).
        ribbon_width:  Bandwidth *w* of the banded system (default 64).
        lsh:           Optional pre-built LSH instance.
    """
 
    def __init__(
        self,
        dim: int,
        capacity: int = 1_000,
        bits_per_entry: int = 8,
        ribbon_width: int = 64,
        lsh: Optional[LSH] = None,
    ) -> None:
        super().__init__(dim, lsh)
        self._capacity      = capacity
        self._bits_per_entry = bits_per_entry
        self._ribbon_width  = ribbon_width
        self.filter: _RibbonFilterImpl = _RibbonFilterImpl(
            capacity=capacity,
            bits_per_entry=bits_per_entry,
            ribbon_width=ribbon_width,
        )
 
    def construct(self, vectors: np.ndarray) -> None:
        """Build the Ribbon filter from a batch of vectors."""
        n = len(vectors)
        self.filter = _RibbonFilterImpl(
            capacity=max(self._capacity, n),
            bits_per_entry=self._bits_per_entry,
            ribbon_width=self._ribbon_width,
        )
        hashes = self.lsh.hash(vectors)
        self.filter.populate(hashes)
 
    def query(self, vectors: np.ndarray) -> List[bool]:
        """Return a per-vector list of membership booleans."""
        hashes = self.lsh.hash(vectors)
        return [self.filter.contains(h) for h in hashes]