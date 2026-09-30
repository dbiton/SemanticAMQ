import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
from filters.filter import Filter


class ConstructInterface(ABC):
    @abstractmethod
    def construct_filter(self, filter: 'Filter', vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_bytesize(self) -> int:
        pass


class InsertInterface(ABC):
    @abstractmethod
    def initialize_buffers(self, filter: 'Filter', spheres_indices: np.ndarray, centers: np.ndarray, radii: np.ndarray,
                           vectors: Optional[np.ndarray] = None) -> None:
        """Called by construct() once the balls are stored. `vectors` are the
        construct vectors themselves -- the only time a strategy sees the
        members of the balls it inherits (for per-ball member counts, or to
        check where the data lies). They are not retained by the filter."""
        pass

    @abstractmethod
    def ensure_capacity(self, required_cap: int) -> None:
        pass

    @abstractmethod
    def reset_buffers(self, capacity: int) -> None:
        pass

    @abstractmethod
    def insert(self, filter: 'Filter', vectors: np.ndarray) -> None:
        pass

    @abstractmethod
    def remove(self, filter: 'Filter', vectors: np.ndarray) -> List[bool]:
        pass

    @abstractmethod
    def get_bytesize(self, filter: 'Filter') -> int:
        pass

    def max_spheres_for_budget(self, usable_bytes: int, bytes_per_ball: int) -> int:
        """How many balls BallCoverFilter may keep when `usable_bytes` (its
        max_memory less the codebook) must hold them *and* this strategy's
        own state at its peak between insert() calls -- so that max_memory
        bounds get_bytesize() over a whole stream, not just after
        construct(). The default suits a strategy with no bookkeeping that
        never holds more than max_spheres balls; one that keeps per-ball
        state, or overshoots max_spheres between rebuilds, charges for it
        here."""
        return usable_bytes // bytes_per_ball