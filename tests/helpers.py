"""Test helpers."""
import os

import atexit
import itertools
import shutil
import tempfile

_LAZY_DIR = tempfile.mkdtemp(prefix="lazy_vectors_")
atexit.register(shutil.rmtree, _LAZY_DIR, ignore_errors=True)
_LAZY_COUNT = itertools.count()


def lazy(array):
    """A VectorSource over a temporary HDF5 file holding `array`: workers take lazy handles only."""
    import h5py
    import numpy as np
    from util.dataset_paths import dataset_filenames, HDF5_DATASET_KEY
    from vector_source import VectorSource
    name = f"_lazy_{next(_LAZY_COUNT)}"
    path = os.path.join(_LAZY_DIR, name + ".h5")
    with h5py.File(path, "w") as f:
        f.create_dataset(HDF5_DATASET_KEY, data=np.asarray(array, dtype=np.float32))
    dataset_filenames[name] = path
    return VectorSource(name)
