import numpy as np
from typing import List, Optional
from filters.filter import Filter
from filters.cap_cover.cap_interfaces import InsertInterface
from filters.cap_cover.cap_geometry import angle_between


class NearestInsert(InsertInterface):
    """Absorb each new vector into the cap whose edge is nearest (least
    angle(x, u) - theta), growing that cap's angle if it does not already
    hold the vector -- never creates caps. Directions are the decoded ones
    query() scores against, so the grown angle is exact."""

    def initialize_buffers(self, filter: Filter, caps_indices: np.ndarray, directions: np.ndarray,
                           angles: np.ndarray, vectors: Optional[np.ndarray] = None) -> None:
        pass

    def ensure_capacity(self, required_cap: int) -> None:
        pass

    def reset_buffers(self, capacity: int) -> None:
        pass

    def insert(self, filter: Filter, vectors: np.ndarray) -> None:
        X = np.asarray(vectors, dtype=np.float64)
        if len(X) == 0 or filter._num_caps == 0:
            return
        ids = filter.all_caps
        D = filter.all_directions
        for s in range(0, len(X), 1024):
            Xb = X[s:s + 1024]
            theta = filter._angle_value(filter._angle_codes[ids])
            # Ranked in float32, decided exactly: the edge-nearest cap by
            # approximate angle, then its exact angle.
            dots = (Xb.astype(np.float32) @ D.astype(np.float32).T).astype(np.float64)
            approx = 2.0 * np.arcsin(np.minimum(1.0, 0.5 * np.sqrt(np.maximum(2.0 - 2.0 * dots, 0.0))))
            col = np.argmin(approx - theta[None, :], axis=1)
            exact = angle_between(Xb, D[col])
            grow = np.full(len(ids), -np.inf)
            np.maximum.at(grow, col, exact)
            hit = np.isfinite(grow)
            filter._grow_angles(ids[hit], grow[hit])

    def remove(self, filter: Filter, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("NearestInsert does not support remove operation.")

    def get_bytesize(self, filter: Filter) -> int:
        return 0
