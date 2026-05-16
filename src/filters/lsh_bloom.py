from datasketch import MinHash, MinHashLSHBloom
from filters.filter import Filter
import numpy as np
from typing import List


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
            fp: Acceptable false positive rate for each Bloom filter in the index.
        """
        super().__init__(dim)
        self.num_perm = num_perm
        # MinHashLSHBloom requires the estimated dataset size (n) and FP rate (fp)
        self.lsh = MinHashLSHBloom(threshold=threshold, num_perm=num_perm, n=n, fp=fp)

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
            self.lsh.insert(f"id_{i}", m)

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