from datasketch import MinHash, MinHashLSHBloom
from filters.filter import Filter
import numpy as np
from scipy.integrate import quad as integrate
from typing import List
import sys


def _false_positive_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - s ** float(r)) ** float(b)
    a, err = integrate(_probability, 0.0, threshold)
    return a, err


def _false_negative_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - (1 - s ** float(r)) ** float(b))
    a, err = integrate(_probability, threshold, 1.0)
    return a, err


def _optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative.

    Returns a tuple: (# bands, band size)
    """
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp, pe = _false_positive_probability(threshold, b, r)
            fn, ne = _false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class LSHBloom(Filter):
    """
    Implementation of LSHBloom using datasketch.MinHashLSHBloom.
    Reference: https://ekzhu.com/datasketch/lshbloom.html
    """
    def __init__(
        self, 
        dim: int, 
        threshold: float = 0.5, 
        num_perm: int = 128, 
        n: int = 1000, 
        fp: float = 0.01,
        lsh = None
    ):
        """
        Args:
            dim: Dimension of the input vectors.
            threshold: Jaccard similarity threshold T.
            num_perm: Number of permutations for MinHash.
            n: Estimated number of sets to be indexed (used to size the Bloom Filter).
            fp: Acceptable total false positive rate for the LSH index.
        """
        super().__init__(dim)
        self.num_perm = num_perm
        
        # Calculate optimal parameters for LSH Bloom filter
        p_effective = fp
        b, r = _optimal_param(threshold, num_perm, 0.5, 0.5)
        
        if b > 0:
            fp_optimal = 1.0 - (1.0 - p_effective)**(1.0 / b)
        else:
            fp_optimal = fp

        self.lsh = MinHashLSHBloom(threshold=threshold, num_perm=num_perm, n=n, fp=fp_optimal)

    def _get_minhash(self, vector: np.ndarray) -> MinHash:
        """Helper to convert a vector into a MinHash signature."""
        m = MinHash(num_perm=self.num_perm)
        for i, val in enumerate(vector):
            # For non-binary vectors, we hash the index and value
            if val != 0:
                m.update(f"{i}:{val}".encode('utf8'))
        return m

    def construct(self, vectors: np.ndarray) -> None:
        """Inserts a batch of vectors into the LSH Bloom Filter."""
        for i, vec in enumerate(vectors):
            if len(vec) != self.dim:
                continue
            m = self._get_minhash(vec)
            # key can be any unique string/identifier
            self.lsh.insert(m)

    def query(self, vectors: np.ndarray) -> List[bool]:
        """
        Checks if a vector with Jaccard similarity >= threshold exists in the filter.
        Returns a list of booleans (True if a duplicate is likely found).
        """
        results = []
        for vec in vectors:
            m = self._get_minhash(vec)
            # .query() returns a binary signal (1 for likely exists, 0 for not)
            is_duplicate = self.lsh.query(m)
            results.append(bool(is_duplicate))
        return results

    def get_bytesize(self) -> int:
        """
        Returns the approximate memory size in bytes of the LSH Bloom Filter.
        
        Estimates based on the MinHash object size and the underlying LSH Bloom structure.
        """
        # Get approximate size from the LSH object
        lsh_size = sys.getsizeof(self.lsh)
        # Add estimated size for MinHash structures (rough estimate)
        # Each MinHash uses num_perm permutations of hash values (4 bytes each)
        minhash_size = self.num_perm * 4  # rough estimate per minhash
        # Estimate total size including internal Bloom filter
        return lsh_size + minhash_size + 2048  # 2KB additional overhead
