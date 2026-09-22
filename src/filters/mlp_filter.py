import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional
from filters.filter import Filter


class BinaryClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MLPFilter(Filter):
    """
    A simple MLP is trained as a binary classifier using the positive
    construction vectors and synthetic samples generated around them. For
    each construction vector x, a random unit direction u is sampled and a
    synthetic point q is generated as:

        q = x + r * u,    where r = alpha * tau

    and alpha is selected from predefined radius factors around the duplicate
    threshold tau. Synthetic samples are labeled using the exact duplicate
    rule: a sample is positive if its nearest construction vector is within
    tau, and negative otherwise.

    The resulting data is used to train a small MLP (input -> hidden -> binary
    output). At query time, the trained classifier predicts the probability
    that a vector is a duplicate, and probabilities >= 0.5 are accepted.
    """
    HIDDEN_DIM = 128
    EPOCHS = 25
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    INNER_RADIUS_RANGE = (0.5, 0.9)
    OUTER_RADIUS_RANGE = (1.1, 1.5)
    SAMPLES_PER_VECTOR = 4
    RANDOM_SEED = 42

    def __init__(self, dim: int, distance_threshold: Optional[float] = None, max_memory: Optional[int] = None):
        super().__init__(dim=dim, max_memory=max_memory)
        self.distance_threshold = distance_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = self._choose_hidden_dim()
        self.model: Optional[BinaryClassifier] = None
        self.rng = np.random.default_rng(self.RANDOM_SEED)
        torch.manual_seed(self.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.RANDOM_SEED)


    def _choose_hidden_dim(self) -> int:
        if self.max_memory is None:
            return self.HIDDEN_DIM

        max_params = self.max_memory // 4
        hidden_dim = (max_params - 1) // (self.dim + 2)
        hidden_dim = min(hidden_dim, self.HIDDEN_DIM)

        if hidden_dim < 1:
            raise ValueError(f"max_memory={self.max_memory} is too small for an MLP with dim={self.dim}")

        return hidden_dim


    def _sample_unit_directions(self, num_samples: int) -> np.ndarray:
        raw = self.rng.standard_normal((num_samples, self.dim), dtype=np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return raw / (norms + 1e-8)


    def _get_radius_factors(self) -> np.ndarray:
        num_samples = self.SAMPLES_PER_VECTOR

        if num_samples == 1:
            return np.array([1.0], dtype=np.float32)

        num_inner = num_samples // 2
        num_outer = num_samples - num_inner

        inner = np.linspace(self.INNER_RADIUS_RANGE[0], self.INNER_RADIUS_RANGE[1], num_inner)
        outer = np.linspace(self.OUTER_RADIUS_RANGE[0], self.OUTER_RADIUS_RANGE[1], num_outer)

        return np.concatenate([inner, outer]).astype(np.float32)


    def _nearest_construct_distances(self, queries: np.ndarray, construct_vectors: np.ndarray) -> np.ndarray:
        construct_norms = np.sum(construct_vectors ** 2, axis=1)
        nearest_distances = np.empty(len(queries), dtype=np.float32)
        batch_size = 1024

        for start in range(0, len(queries), batch_size):
            end = min(start + batch_size, len(queries))
            batch = queries[start:end]
            batch_norms = np.sum(batch ** 2, axis=1)
            squared_distances = batch_norms[:, None] + construct_norms[None, :] - 2.0 * (batch @ construct_vectors.T)
            nearest_distances[start:end] = np.sqrt(np.maximum(np.min(squared_distances, axis=1), 0.0))

        return nearest_distances


    def _generate_training_data(self, vectors: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
        num_positive = len(vectors)
        radius_factors = self._get_radius_factors()
        num_samples = num_positive * self.SAMPLES_PER_VECTOR

        base_vectors = np.repeat(vectors, self.SAMPLES_PER_VECTOR, axis=0)
        factors = np.tile(radius_factors, num_positive)

        directions = self._sample_unit_directions(num_samples)
        radii = (factors * tau).astype(np.float32).reshape(-1, 1)
        synthetic_points = base_vectors + directions * radii

        X = np.vstack([vectors, synthetic_points]).astype(np.float32)

        labels = np.zeros(len(X), dtype=np.float32)
        labels[:num_positive] = 1.0

        nearest_distances = self._nearest_construct_distances(synthetic_points, vectors)
        labels[num_positive:] = (nearest_distances <= tau).astype(np.float32)

        return X, labels


    def _train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model = BinaryClassifier(self.dim, self.hidden_dim).to(self.device)

        X_tensor = torch.from_numpy(X).to(self.device)
        y_tensor = torch.from_numpy(y).unsqueeze(1).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        criterion = nn.BCELoss()

        self.model.train()

        for _ in range(self.EPOCHS):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

        self.model.eval()


    def construct(self, vectors: np.ndarray) -> None:
        if len(vectors) == 0:
            return

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        X, y = self._generate_training_data(vectors, self.distance_threshold)
        self._train(X, y)


    def query(self, vectors: np.ndarray) -> List[bool]:
        if self.model is None:
            raise ValueError("MLPFilter has not been constructed.")

        if len(vectors) == 0:
            return [False] * len(vectors)

        X_tensor = torch.from_numpy(np.ascontiguousarray(vectors, dtype=np.float32)).to(self.device)

        with torch.no_grad():
            probs = self.model(X_tensor).squeeze(1).cpu().numpy()

        return (probs >= 0.5).tolist()


    def insert(self, vectors: np.ndarray) -> None:
        raise NotImplementedError("MLPFilter does not support insert operation.")


    def add(self, vectors: np.ndarray) -> None:
        self.insert(vectors)


    def remove(self, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("MLPFilter does not support remove operation.")


    def get_bytesize(self) -> int:
        if self.model is None:
            return 0
        model_bytes = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_bytes = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        return model_bytes + buffer_bytes