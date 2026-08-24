import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List
from filters.filter import Filter


class ConstructInterface(ABC):
    @abstractmethod
    def construct_filter(self, filter: 'Filter', vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Builds sphere centers and radii covering the input vector dataset."""
        pass

    @abstractmethod
    def get_bytesize(self) -> int:
        """Returns the memory overhead of temporary construction structures."""
        pass


class InsertInterface(ABC):
    @abstractmethod
    def ensure_capacity(self, required_cap: int) -> None:
        """Resizes internal tracking arrays when the filter expands its buffer capacity."""
        pass

    @abstractmethod
    def reset_buffers(self, capacity: int) -> None:
        """Clears state and aligns tracking buffer size with filter capacity."""
        pass

    @abstractmethod
    def add_sphere(self, filter: 'Filter', center: np.ndarray, radius: float) -> None:
        """Places a new sphere into the filter and updates candidate merge structures."""
        pass

    @abstractmethod
    def add(self, filter: 'Filter', vectors: np.ndarray) -> None:
        """Filters out covered points and dynamically inserts uncovered vectors."""
        pass

    def insert(self, filter: 'Filter', vectors: np.ndarray) -> None:
        """Delegates directly to `add` for vector insertion."""
        self.add(filter, vectors)

    @abstractmethod
    def remove(self, filter: 'Filter', vectors: np.ndarray) -> List[bool]:
        """Deletes vectors from the active filter state and returns removal status."""
        pass

    @abstractmethod
    def get_bytesize(self) -> int:
        """Returns memory consumed by internal insertion tracking structures."""
        pass