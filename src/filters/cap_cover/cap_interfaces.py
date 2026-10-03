import numpy as np
from abc import ABC, abstractmethod
from typing import Iterator, NamedTuple, Tuple, List, Optional
from filters.filter import Filter
from filters.cap_cover.cap_geometry import labelled_cap_cover


class Cover(NamedTuple):
    """Caps (directions[j], angles[j]) covering a set of unit rows: row i
    lies in cap labels[i] within its angle. neighbors[j] (-1 padded) lists
    caps near cap j: where the filter looks for a member's nearest caps
    instead of searching them all. None: search them all."""
    directions: np.ndarray
    angles: np.ndarray
    labels: np.ndarray
    neighbors: Optional[np.ndarray] = None


class ConstructInterface(ABC):
    @abstractmethod
    def construct_filter(self, filter: 'Filter', vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(directions, angles): unit cap directions and their angular radii,
        covering every row of `vectors` (unit rows, in the filter's working
        basis). The filter re-measures every angle after encoding, so an
        angle here only has to be an honest cover of the *unencoded*
        directions."""
        pass

    def iter_candidates(self, filter: 'Filter', vectors: np.ndarray, ks: List[int],
                        weights: Optional[np.ndarray] = None) -> Iterator[Tuple[int, Cover]]:
        """(k, cover) for each cap count in `ks`, largest first, for a
        filter choosing among encodings of different sizes. `vectors` are
        distinct and `weights` (default 1) counts the copies of each, for a
        method that weighs members. Candidates may be cheaper than
        construct_filter's covers (unrefined); finish() turns the chosen one
        into a final cover. The default builds each from scratch with
        construct_filter; a method with a natural hierarchy
        (SphereMergeConstruct) overrides this to share the work."""
        saved = filter.max_construct_caps
        try:
            for k in sorted(set(int(k) for k in ks), reverse=True):
                filter.max_construct_caps = k
                U, _ = self.construct_filter(filter, vectors)
                if len(U) == 0:
                    yield k, Cover(U, np.empty(0), np.zeros(len(vectors), dtype=np.int64))
                    continue
                U, theta, labels, _ = labelled_cap_cover(vectors, U)
                yield k, Cover(U, theta, labels)
        finally:
            filter.max_construct_caps = saved

    def finish(self, filter: 'Filter', vectors: np.ndarray, cover: Cover) -> Cover:
        """The final cover from a candidate iter_candidates() produced."""
        return cover

    @abstractmethod
    def get_bytesize(self) -> int:
        pass


class InsertInterface(ABC):
    @abstractmethod
    def initialize_buffers(self, filter: 'Filter', caps_indices: np.ndarray, directions: np.ndarray,
                           angles: np.ndarray, vectors: Optional[np.ndarray] = None) -> None:
        """Called by construct() once the caps are stored. `directions` are the
        decoded unit directions query() scores against and `vectors` the
        construct vectors (unit rows, working basis) -- the only time a
        strategy sees the members of the caps it inherits. Not retained by
        the filter."""
        pass

    @abstractmethod
    def ensure_capacity(self, required_cap: int) -> None:
        pass

    @abstractmethod
    def reset_buffers(self, capacity: int) -> None:
        pass

    @abstractmethod
    def insert(self, filter: 'Filter', vectors: np.ndarray) -> None:
        """`vectors` are unit rows in the filter's working basis."""
        pass

    @abstractmethod
    def remove(self, filter: 'Filter', vectors: np.ndarray) -> List[bool]:
        pass

    @abstractmethod
    def get_bytesize(self, filter: 'Filter') -> int:
        pass

    def max_caps_for_budget(self, usable_bytes: int, bytes_per_cap: int) -> int:
        """How many caps CapCoverFilter may keep when `usable_bytes` must hold
        them and this strategy's own state at its peak between insert()
        calls (BallCoverFilter's InsertInterface.max_spheres_for_budget)."""
        return usable_bytes // bytes_per_cap
