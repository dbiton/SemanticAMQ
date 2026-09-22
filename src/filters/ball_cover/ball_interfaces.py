import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List
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
    def initialize_buffers(self, filter: 'Filter', spheres_indices: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> None:
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