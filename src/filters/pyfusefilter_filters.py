import numpy as np
from lsh import LSH
from typing import Optional, List
from filters.filter import Filter
from pyfusefilter import Xor8, Xor16, Fuse8, Fuse16


class PyFuseFilter(Filter):
    def __init__(self, dim: int, lsh: Optional[LSH] = None):
        super().__init__(dim, lsh)
        self.filter: Xor8 | Xor16 | Fuse8 | Fuse16


class Xor8Filter(PyFuseFilter):
    def __init__(self, dim: int, lsh: Optional[LSH] = None):
        super().__init__(dim, lsh)
        self.filter = Xor8(dim)

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

    def construct(self, vectors: np.ndarray) -> None:
        hashes = self.lsh.hash(vectors)
        self.filter.populate(hashes)

    def query(self, vectors: np.ndarray) -> List[bool]:
        hash = self.lsh.hash(vectors.reshape(1, -1))[0]
        return self.filter.contains(hash)


class Fuse16Filter(PyFuseFilter):
    def __init__(self, dim: int, lsh: Optional[LSH] = None):
        super().__init__(dim, lsh)
        self.filter = Fuse16(dim)

    def construct(self, vectors: np.ndarray) -> None:
        hashes = self.lsh.hash(vectors)
        self.filter.populate(hashes)

    def query(self, vectors: np.ndarray) -> List[bool]:
        hashes = self.lsh.hash(vectors)
        return [self.filter.contains(hash) for hash in hashes]
