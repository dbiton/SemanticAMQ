from typing import List, Tuple

import faiss
import numpy as np


class BallCoverFilter:
    def __init__(self, dim: int, max_memory: int):
        self.index = faiss.IndexFlatL2(dim)
        self.dim = dim
        self.max_count_balls = self.calculate_max_balls(max_memory)
        self.count_balls = 0
        self.radii = np.zeros(self.max_count_balls, dtype=np.float32)

    def bytes_per_ball(self) -> int:
        return (self.index.d + 1) * 4

    def calculate_max_balls(self, max_memory: int) -> int:
        return max(1, max_memory // self.bytes_per_ball())

    def get_bytesize(self) -> int:
        return self.count_balls * self.bytes_per_ball()

    def construct_enclosing_balls(
        self, vectors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gonzalez farthest-first, then exact radii. O(n*k*d)."""
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(X)
        k = min(self.max_count_balls, n)
        if n <= k:
            return X.copy(), np.zeros(n, dtype=np.float32)

        chosen = [0]
        min_d2 = ((X - X[0]) ** 2).sum(1)
        for _ in range(k - 1):
            chosen.append(int(min_d2.argmax()))
            min_d2 = np.minimum(min_d2, ((X - X[chosen[-1]]) ** 2).sum(1))
        centers = np.ascontiguousarray(X[chosen])

        index = faiss.IndexFlatL2(self.dim)
        index.add(centers)
        d2, labels = index.search(X, 1)
        radii = np.zeros(k, dtype=np.float32)
        np.maximum.at(radii, labels.ravel(), np.sqrt(np.maximum(d2.ravel(), 0)))
        return centers, radii

    def construct(self, vectors: np.ndarray) -> None:
        centers, radii = self.construct_enclosing_balls(vectors)
        n = len(centers)
        self.index.reset()
        self.index.add(np.ascontiguousarray(centers, dtype=np.float32))
        self.radii[:] = 0.0
        self.radii[:n] = radii
        self.count_balls = n

    def distance_to_balls_surfaces(self, query_vectors: np.ndarray) -> np.ndarray:
        """(nq, k) signed distance to each ball surface. Negative = inside."""
        Q = np.ascontiguousarray(query_vectors, dtype=np.float32)
        d2, labels = self.index.search(Q, self.count_balls)
        return np.sqrt(d2) - self.radii[labels]

    def query(self, query_vectors: np.ndarray, distance_threshold: float) -> List[bool]:
        surface = self.distance_to_balls_surfaces(query_vectors)
        return (surface.min(axis=1) < distance_threshold).tolist()

    def add(self, vectors: np.ndarray) -> None:
        """Absorb each vector into one existing ball, inflating it if needed."""
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        d2, labels = self.index.search(X, self.count_balls)
        dist = np.sqrt(np.maximum(d2, 0.0))
        surface = dist - self.radii[labels]

        cost = np.where(surface <= 0, -np.inf, dist)
        col = cost.argmin(axis=1)
        row = np.arange(len(X))
        ball = labels[row, col]
        np.maximum.at(self.radii, ball, dist[row, col])