from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import h5py
import numpy as np

# Import from the lightweight registry (no torch / sentence_transformers /
# datasets) so spawned workers that only read vectors don't drag in the whole
# ML stack (~400 MB RSS per process).
from util.dataset_paths import dataset_filenames, HDF5_DATASET_KEY as _HDF5_DATASET_KEY


@dataclass(frozen=True)
class VectorSource:
    """
    A lightweight, picklable pointer to a contiguous slice of a dataset's
    vectors stored on disk.

    It holds *no* vector data itself -- only the dataset name and slice bounds.
    Vectors are streamed lazily from the backing HDF5 file on demand, one disk
    chunk at a time, so handing a VectorSource to a job costs O(1) memory
    regardless of how many vectors the slice covers.

    This is what lets the query stream grow arbitrarily long: previously the
    whole array was copied into every job (and pickled to every worker), so
    memory scaled with num_jobs * stream_length. A VectorSource scales with 1.
    """

    dataset_name: str
    start: int = 0
    stop: Optional[int] = None
    chunk_size: int = 4096

    def _resolve_stop(self, dset_len: int) -> int:
        return dset_len if self.stop is None else min(self.stop, dset_len)

    def __len__(self) -> int:
        path = dataset_filenames[self.dataset_name]
        with h5py.File(path, "r") as f:
            dset_len = f[_HDF5_DATASET_KEY].shape[0]
        return max(0, self._resolve_stop(dset_len) - self.start)

    def iter_batches(self, batch_size: Optional[int] = None) -> Iterator[np.ndarray]:
        """
        Yield the slice as consecutive 2D float32 blocks, reading one disk chunk
        at a time. Never materialises more than `batch_size` rows at once, so an
        arbitrarily long stream is processed in bounded memory.
        """
        batch_size = batch_size or self.chunk_size
        path = dataset_filenames[self.dataset_name]
        with h5py.File(path, "r") as f:
            dset = f[_HDF5_DATASET_KEY]
            stop = self._resolve_stop(dset.shape[0])
            for base in range(self.start, stop, batch_size):
                end = min(base + batch_size, stop)
                yield np.asarray(dset[base:end], dtype=np.float32)

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Yield the slice one vector (1D float32 array) at a time. Disk reads are
        still chunked underneath, so per-vector iteration stays fast.
        """
        for block in self.iter_batches():
            for vector in block:
                yield vector

    def to_array(self) -> np.ndarray:
        """
        Materialise the whole slice as a single 2D array.

        Use only where a batch is genuinely required at once (e.g. building a
        static/immutable filter from all keys). For the query stream, prefer
        iter_batches() / iteration so nothing larger than a chunk is held.
        """
        path = dataset_filenames[self.dataset_name]
        with h5py.File(path, "r") as f:
            dset = f[_HDF5_DATASET_KEY]
            stop = self._resolve_stop(dset.shape[0])
            return np.asarray(dset[self.start:stop], dtype=np.float32)
