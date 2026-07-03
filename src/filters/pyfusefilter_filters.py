import numpy as np
from lsh import LSH
from typing import Optional, List, Tuple
from filters.filter import Filter
from pyfusefilter import Xor8, Xor16, Fuse8, Fuse16


class PyFuseFilter(Filter):
    def __init__(self, dim: int, lsh: Optional[LSH] = None, max_memory: Optional[int] = None):
        super().__init__(dim, lsh, max_memory=max_memory)
        self.filter: Xor8 | Xor16 | Fuse8 | Fuse16
        self.epsilon: float = 0.0  # Overridden by subclasses
        self._n_keys: int = 0

    def get_exact_matching_fpr(self) -> float:
        """
        Returns the standard exact-matching False Positive Rate (FPR) 
        of the underlying AMQ structure, completely ignoring the LSH layer.
        """
        return float(self.epsilon)

    def add(self, vectors: np.ndarray) -> None:
        """
        PyFuseFilters are static, immutable structures and do not support dynamic insertion.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support dynamic insertion. "
            "PyFuseFilters are static and immutable. Please reconstruct with the full dataset."
        )

    def remove(self, vectors: np.ndarray) -> List[bool]:
        """
        PyFuseFilters are static, immutable structures and do not support removal.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support removal. "
            "PyFuseFilters are static and immutable. Please reconstruct with the full dataset."
        )

    def get_bytesize(self) -> int:
        """
        Returns the memory size in bytes of the underlying fingerprint table.

        Uses the library's own ``size_in_bytes()`` (the exact size of the
        fingerprint array plus its small header) rather than a hand-tuned
        slots-per-key factor, which previously under-counted the Fuse variants
        by ~14%.
        """
        return int(self.filter.size_in_bytes())

    def _enforce_memory_budget(self) -> None:
        """
        Raise if the just-constructed filter exceeds max_memory.

        PyFuseFilters are static and size purely from the number of keys, so
        the only way to honour a memory budget is to refuse to build a filter
        that would exceed it.
        """
        if self.max_memory is not None and self.get_bytesize() > self.max_memory:
            raise ValueError(
                f"{self.__class__.__name__}: {self._n_keys} keys require "
                f"{self.get_bytesize()} bytes, exceeding max_memory={self.max_memory}."
            )


class Xor8Filter(PyFuseFilter):
    def __init__(self, dim: int, lsh: Optional[LSH] = None, max_memory: Optional[int] = None):
        super().__init__(dim, lsh, max_memory=max_memory)
        self.filter = Xor8(0)
        self.epsilon = 1.0 / 256.0  # 8-bit fingerprint error rate (1/2^8)

    def construct(self, vectors: np.ndarray):
        hashes = self._normalize_hashes(self.lsh.hash(vectors))
        signature_keys = [key for signature_keys in hashes for key in signature_keys]
        self._n_keys = len(signature_keys)
        self.filter = Xor8(size_or_data=signature_keys)
        self._enforce_memory_budget()

    def query(self, vectors: np.ndarray):
        hashes = self._normalize_hashes(self.lsh.hash(vectors))
        return [any(self.filter.contains(key) for key in signature_keys) for signature_keys in hashes]


class Xor16Filter(PyFuseFilter):
    def __init__(self, dim: int, lsh: Optional[LSH] = None, max_memory: Optional[int] = None):
        super().__init__(dim, lsh, max_memory=max_memory)
        self.filter = Xor16(0)
        self.epsilon = 1.0 / 65536.0  # 16-bit fingerprint error rate (1/2^16)

    def construct(self, vectors: np.ndarray) -> None:
        hashes = self._normalize_hashes(self.lsh.hash(vectors))
        signature_keys = [key for signature_keys in hashes for key in signature_keys]
        self._n_keys = len(signature_keys)
        self.filter = Xor16(size_or_data=signature_keys)
        self._enforce_memory_budget()

    def query(self, vectors: np.ndarray) -> List[bool]:
        hashes = self._normalize_hashes(self.lsh.hash(vectors))
        return [any(self.filter.contains(key) for key in signature_keys) for signature_keys in hashes]


class Fuse8Filter(PyFuseFilter):
    def __init__(self, dim: int, lsh: Optional[LSH] = None, max_memory: Optional[int] = None):
        super().__init__(dim, lsh, max_memory=max_memory)
        self.filter = Fuse8(0)
        self.epsilon = 1.0 / 256.0  # 8-bit fingerprint error rate (1/2^8)

    def construct(self, vectors: np.ndarray) -> None:
        hashes = self._normalize_hashes(self.lsh.hash(vectors))
        signature_keys = [key for signature_keys in hashes for key in signature_keys]
        self._n_keys = len(signature_keys)
        self.filter = Fuse8(size_or_data=signature_keys)
        self._enforce_memory_budget()

    def query(self, vectors: np.ndarray) -> List[bool]:
        hashes = self._normalize_hashes(self.lsh.hash(vectors))
        return [any(self.filter.contains(key) for key in signature_keys) for signature_keys in hashes]


class Fuse16Filter(PyFuseFilter):
    def __init__(self, dim: int, lsh: Optional[LSH] = None, max_memory: Optional[int] = None):
        super().__init__(dim, lsh, max_memory=max_memory)
        self.filter = Fuse16(0)
        self.epsilon = 1.0 / 65536.0  # 16-bit fingerprint error rate (1/2^16)

    def construct(self, vectors: np.ndarray) -> None:
        hashes = self._normalize_hashes(self.lsh.hash(vectors))
        signature_keys = [key for signature_keys in hashes for key in signature_keys]
        self._n_keys = len(signature_keys)
        self.filter = Fuse16(size_or_data=signature_keys)
        self._enforce_memory_budget()

    def query(self, vectors: np.ndarray) -> List[bool]:
        hashes = self._normalize_hashes(self.lsh.hash(vectors))
        return [any(self.filter.contains(key) for key in signature_keys) for signature_keys in hashes]