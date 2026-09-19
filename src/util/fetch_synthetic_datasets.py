import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import h5py
import numpy as np
from typing import Tuple, List


def generate_synthetic_dataset(
    num_construct: int,
    num_query: int,
    dim: int = 16,
    target_vol: float = 0.1,
    max_boxes: int = 10,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates axis-aligned boxes occupying `target_vol` cumulative fraction of
    a [0, 1]^D unit hypercube. construct_vectors are sampled proportionally to
    box volumes (positive samples); query_vectors are uniform over the whole
    hypercube. Returns (construct_vectors, query_vectors, boxes_d, boxes_a) --
    boxes_d/boxes_a are each box's side lengths and lower-bound origin.
    """
    rng = np.random.default_rng(seed)

    boxes_d, boxes_a = [], []
    s = target_vol

    print("Generating", dim, "dimensional boxes to fill", target_vol * 100, "% of the space...")

    while s > 1e-9 and len(boxes_a) < max_boxes:
        if len(boxes_a) == max_boxes - 1:
            # Last box: force its volume to exactly consume remaining s.
            w = rng.uniform(0.5, 1.0, size=dim)
            d = s ** (w / np.sum(w))
            s = 0.0
        else:
            upper_bound = s ** (1.0 / dim)
            d = rng.uniform(upper_bound * 0.5, upper_bound, size=dim)
            s -= np.prod(d)

        a = rng.uniform(0, 1 - d)

        boxes_d.append(d)
        boxes_a.append(a)

    print(f"Successfully generated {len(boxes_a)} boxes.")
    print(f"Generating {num_construct} construct vectors...")

    boxes_d_arr, boxes_a_arr = np.array(boxes_d, dtype=np.float32), np.array(boxes_a, dtype=np.float32)
    box_volumes = np.prod(boxes_d_arr, axis=1)
    box_probabilities = box_volumes / box_volumes.sum()

    box_indices = rng.choice(len(boxes_a), size=num_construct, p=box_probabilities)
    construct_d, construct_a = boxes_d_arr[box_indices], boxes_a_arr[box_indices]

    # a + U[0,1]*d guarantees the point is strictly inside the chosen box.
    construct_vectors = construct_a + rng.uniform(0, 1, size=(num_construct, dim)) * construct_d
    construct_vectors = construct_vectors.astype(np.float32)

    print(f"Generating {num_query} query vectors...")
    query_vectors = rng.uniform(0, 1, size=(num_query, dim)).astype(np.float32)

    return construct_vectors, query_vectors, boxes_d_arr, boxes_a_arr


def save_synthetic_dataset(
    filename: str = "synthetic_d16", 
    dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = None
) -> None:
    """Saves generated synthetic dataset to an HDF5 file."""
    filepath = f"./datasets/{filename}.h5"
    construct_vectors, query_vectors, boxes_d, boxes_a = dataset

    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with h5py.File(filepath, 'w') as f:
        f.create_dataset("construct_vectors", data=construct_vectors, dtype=np.float32)
        f.create_dataset("query_vectors", data=query_vectors, dtype=np.float32)
        f.create_dataset("boxes_d", data=boxes_d, dtype=np.float32)
        f.create_dataset("boxes_a", data=boxes_a, dtype=np.float32)
    print(f"Dataset successfully saved to {filepath}")


def load_synthetic_dataset(
    filename: str = "synthetic_d16"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """Loads a synthetic dataset from an HDF5 file."""
    filepath = f"./datasets/{filename}.h5"
    print(f"Loading synthetic dataset from {filepath}...")
    
    with h5py.File(filepath, 'r') as f:
        construct_vectors = f["construct_vectors"][:]
        query_vectors = f["query_vectors"][:]
        boxes_d = f["boxes_d"][:]
        boxes_a = f["boxes_a"][:]

    return construct_vectors, query_vectors, boxes_d, boxes_a


if __name__ == "__main__":
    dataset_name         = "synthetic_d16"
    num_vectors          = 100000
    percentage_construct = 10
    dim                  = 16
    target_vol           = 0.1
    max_boxes            = 10
    seed                 = 42

    num_construct = num_vectors * percentage_construct // 100
    num_query = num_vectors - num_construct

    print(f"Generating synthetic dataset num_construct={num_construct}, num_query={num_query}, dim={dim} ...")
    dataset = generate_synthetic_dataset(num_construct, num_query, dim, target_vol, max_boxes, seed)
    save_synthetic_dataset(dataset_name, dataset)
    print("Generated synthetic dataset successfully.")