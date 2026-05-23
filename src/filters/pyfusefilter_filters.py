import numpy as np
import sys
from lsh import LSH
from typing import Optional, List, Tuple
from filters.filter import Filter
from pyfusefilter import Xor8, Xor16, Fuse8, Fuse16


class PyFuseFilter(Filter):
    def __init__(self, dim: int, lsh: Optional[LSH] = None):
        super().__init__(dim, lsh)
        self.filter: Xor8 | Xor16 | Fuse8 | Fuse16
        self.epsilon: float = 0.0  # Overridden by subclasses

    def get_exact_matching_fpr(self) -> float:
        """
        Returns the standard exact-matching False Positive Rate (FPR) 
        of the underlying AMQ structure, completely ignoring the LSH layer.
        """
        return float(self.epsilon)

    def get_bytesize(self) -> int:
        """
        Returns the approximate memory size in bytes of the PyFuseFilter.
        
        This is a base implementation; subclasses may override for specific estimates.
        """
        return sys.getsizeof(self.filter) + 512  # 512 bytes overhead


class Xor8Filter(PyFuseFilter):
    def __init__(self, dim: int, lsh: Optional[LSH] = None):
        super().__init__(dim, lsh)
        self.filter = Xor8(dim)
        self.epsilon = 1.0 / 256.0  # 8-bit fingerprint error rate (1/2^8)

    def get_bytesize(self) -> int:
        """
        Returns the approximate memory size in bytes of the Xor8 filter.
        
        Xor8 filters use roughly 8 bits per element.
        """
        return (self.dim * 8) // 8 + 512  # 8 bits per element converted to bytes, plus overhead

    def construct(self, vectors: np.ndarray):
        hashes = self.lsh.hash(vectors)
        self.filter.populate(hashes)

    def query(self, vectors: np.ndarray):
        hashes = self.lsh.hash(vectors)
        return [self.filter.contains(hash) for hash in hashes]


class Xor16Filter(PyFuseFilter):
    def __init__(self, dim: int, lsh: Optional[LSH] = None):
        super().__init__(dim, lsh)
        self.filter = Xor16(dim)
        self.epsilon = 1.0 / 65536.0  # 16-bit fingerprint error rate (1/2^16)

    def get_bytesize(self) -> int:
        """
        Returns the approximate memory size in bytes of the Xor16 filter.
        
        Xor16 filters use roughly 16 bits per element.
        """
        return (self.dim * 16) // 8 + 512  # 16 bits per element converted to bytes, plus overhead

    def construct(self, vectors: np.ndarray) -> None:
        hashes = self.lsh.hash(vectors)
        self.filter.populate(hashes)

    def query(self, vectors: np.ndarray) -> List[bool]:
        hashes = self.lsh.hash(vectors)
        return [self.filter.contains(hash) for hash in hashes]


class Fuse8Filter(PyFuseFilter):
    def __init__(self, dim: int, lsh: Optional[LSH] = None):
        super().__init__(dim, lsh)
        self.filter = Fuse8(dim)
        self.epsilon = 1.0 / 256.0  # 8-bit fingerprint error rate (1/2^8)

    def get_bytesize(self) -> int:
        """
        Returns the approximate memory size in bytes of the Fuse8 filter.
        
        Fuse8 filters use roughly 8 bits per element.
        """
        return (self.dim * 8) // 8 + 512  # 8 bits per element converted to bytes, plus overhead

    def construct(self, vectors: np.ndarray) -> None:
        hashes = self.lsh.hash(vectors)
        self.filter.populate(hashes)

    def query(self, vectors: np.ndarray) -> List[bool]:
        hashes = self.lsh.hash(vectors)
        return [self.filter.contains(hash) for hash in hashes]


class Fuse16Filter(PyFuseFilter):
    def __init__(self, dim: int, lsh: Optional[LSH] = None):
        super().__init__(dim, lsh)
        self.filter = Fuse16(dim)
        self.epsilon = 1.0 / 65536.0  # 16-bit fingerprint error rate (1/2^16)

    def get_bytesize(self) -> int:
        """
        Returns the approximate memory size in bytes of the Fuse16 filter.
        
        Fuse16 filters use roughly 16 bits per element.
        """
        return (self.dim * 16) // 8 + 512  # 16 bits per element converted to bytes, plus overhead

    def construct(self, vectors: np.ndarray) -> None:
        hashes = self.lsh.hash(vectors)
        self.filter.populate(hashes)

    def query(self, vectors: np.ndarray) -> List[bool]:
        hashes = self.lsh.hash(vectors)
        return [self.filter.contains(hash) for hash in hashes]