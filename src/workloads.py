"""Workloads: positional fold selection over a corpus's HDF5 file, exact ground truth by distance
at tau, and the stratified calibration split. Explicit inputs only -- no evaluation settings;
experiments.py supplies the sizes, folds and corpora."""
import logging
import os
from typing import Dict, List, Optional, Tuple

# Before numpy/faiss are imported (see experiments.py): one thread per worker process.
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import faiss
import h5py
import numpy as np

from util.dataset_paths import dataset_filenames, HDF5_DATASET_KEY
from vector_source import VectorSource

logger = logging.getLogger(__name__)





_READ_WINDOW = 65_536  # rows per contiguous HDF5 read in _read_rows (~100 MB at d=384)


def _read_rows(dataset: str, idx: np.ndarray) -> np.ndarray:
    """Just the requested rows of a dataset's embeddings, straight from HDF5.

    Read as contiguous windows of _READ_WINDOW rows, keeping the requested rows
    of each, rather than by h5py fancy indexing: that builds one selection per
    point and scales superlinearly -- 582s for a fold's 80K scattered rows
    against 1.6s for all 808K Quora rows read contiguously. The request is
    sorted for the read and then mapped back to the caller's order.
    """
    order = np.argsort(idx, kind="stable")
    sorted_idx = idx[order]
    uniq, inverse = np.unique(sorted_idx, return_inverse=True)
    with h5py.File(dataset_filenames[dataset], "r") as f:
        dset = f[HDF5_DATASET_KEY]
        block = np.empty((len(uniq), dset.shape[1]), dtype=np.float32)
        a = 0
        while a < len(uniq):
            lo = int(uniq[a])
            b = int(np.searchsorted(uniq, lo + _READ_WINDOW))
            block[a:b] = dset[lo:int(uniq[b - 1]) + 1][uniq[a:b] - lo]
            a = b
    out = np.empty((len(idx), block.shape[1]), dtype=np.float32)
    out[order] = block[inverse]
    return np.ascontiguousarray(out)


class BlocklistWorkload:
    """One corpus's blocklist workload: cluster labels, the per-fold
    reference/query index selection, and a reader for the rows it names.

    Shared by generate_blocklist and generate_quantization so both evaluate on
    the same construction and query sets. The two then differ only in what
    they sweep -- ball budget versus quantizer -- which is the point: a
    difference between their results is a difference in the swept variable,
    not in the data underneath.
    """

    def __init__(self, name: str, plot_key: str, selections: List[Dict[str, np.ndarray]],
                 pool: Optional[np.ndarray], stream_size: int, reference_target: int) -> None:
        self.name = name
        self.plot_key = plot_key
        self.selections = selections
        self._pool = pool
        self.stream_size = stream_size
        self.reference_target = reference_target

    def rows(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        return self._pool[idx] if self._pool is not None else _read_rows(self.name, idx)

    def query_vectors(self, sel: Dict[str, np.ndarray]) -> np.ndarray:
        """A fold's queries as one array. Ground truth is always distance to
        the reference set at tau (compute_static_actual_results); how related
        a query is to the set (runner.SIMILARITY_BANDS) only groups FPR."""
        return self.rows(sel["queries"])


def build_blocklist_workload(name: str, plot_key: str, num_vectors: int, n_folds: int,
                             query_fraction: float = 0.2) -> Optional[BlocklistWorkload]:
    """Cut a corpus into per-fold construct/query splits by file position, or
    return None (logging why) if the file is too short.

    No clustering and no sampling: fold f constructs from the f-th window of
    the first rows of the file and queries from the f-th window of its last
    rows, so the two regions never overlap and folds are disjoint. All
    queries are carried as one stream; strata are not defined by file
    position, and ground truth is distance at tau regardless."""
    if not os.path.exists(dataset_filenames[name]):
        logger.warning(f"Skipping {name}: not found on disk")
        return None

    total = len(VectorSource(name))
    stream_size = int(round(query_fraction * num_vectors))
    reference_target = num_vectors - stream_size
    if n_folds * (reference_target + stream_size) > total:
        logger.error(
            f"Skipping {name}: {n_folds} folds of {reference_target} construct + {stream_size} "
            f"query rows need {n_folds * num_vectors} rows but the file has {total}.")
        return None

    selections = [{
        "reference": np.arange(f * reference_target, (f + 1) * reference_target, dtype=np.int64),
        "queries": np.arange(total - (f + 1) * stream_size, total - f * stream_size, dtype=np.int64),
    } for f in range(n_folds)]

    logger.info(f"{name}: |B|={reference_target} from the start of the file, "
                f"stream={stream_size} from the end ({total} rows)")
    return BlocklistWorkload(name, plot_key, selections, None, stream_size, reference_target)



def compute_static_actual_results(
    construct_vectors: np.ndarray,
    query_vectors: np.ndarray,
    distance_threshold: float,
) -> list[bool]:
    """Static ground-truth: a query vector is a duplicate iff within
    `distance_threshold` of any vector in the fixed construct set. Uses one
    batched faiss nearest-neighbor call rather than a per-query Python loop;
    faiss returns squared L2, hence the distance_threshold**2 comparison."""
    construct = np.ascontiguousarray(construct_vectors, dtype=np.float32)
    queries = np.ascontiguousarray(query_vectors, dtype=np.float32)

    if len(construct) == 0:
        return [False] * len(queries)

    index = faiss.IndexFlatL2(construct.shape[1])
    index.add(construct)
    nn_sq_dist, _ = index.search(queries, 1)
    return (nn_sq_dist[:, 0] <= distance_threshold ** 2).tolist()


def stratified_calibration_split(
    stream: np.ndarray,
    actual: List[bool],
    calibration_fraction: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[bool], List[bool]]:
    """Cuts a query stream into a calibration slice and a disjoint measurement
    slice that carry the SAME duplicate rate, and returns
    (calibration_vectors, measurement_vectors, calibration_actual,
    measurement_actual).

    Stratified on `actual`, and shuffled within each stratum, because a
    prefix slice of a file-ordered stream is not a sample of it (earlier, the
    stream was a concatenation of query sources and the prefix was exactly
    one of them), so:

      - calibration saw a set that was ~91% duplicates on Quora, where F1 is
        maximised by answering "duplicate" unconditionally -- which is the
        config _build_calibrated_filter then picked, and the reason filters
        reported FPR near 1.0 while their FNR was 0.
      - measurement saw the negatives only: 2 of 16,000 Quora queries were
        within tau of the construct set, 0 of 16,000 on WikiAnswers and
        ChatGPTParaphrases. Every reported FNR was therefore either a
        2-sample estimate or the fn/(fn+tp)==0/0 fallback of 0.0, not a
        measurement -- IVFFilter's "FNR 1.000" was it missing both of Quora's
        two positives.

    Stratifying keeps both slices representative of the whole stream, so the
    dial chosen on one is the dial that should win on the other."""
    stream = np.asarray(stream)
    labels = np.asarray(actual, dtype=bool)
    if len(stream) != len(labels):
        raise ValueError(
            f"stratified_calibration_split: {len(stream)} vectors but {len(labels)} labels.")

    rng = np.random.default_rng(seed)
    calibration_idx, measurement_idx = [], []
    for value in (True, False):
        group = np.flatnonzero(labels == value)
        if len(group) == 0:
            continue
        rng.shuffle(group)
        # round(), not floor, so a small positive stratum still lands in
        # calibration; len(group)-1 so a stratum is never consumed whole,
        # leaving the measurement slice without it.
        n_calibration = 0
        if len(group) > 1:
            n_calibration = min(len(group) - 1,
                                max(1, round(len(group) * calibration_fraction)))
        calibration_idx.append(group[:n_calibration])
        measurement_idx.append(group[n_calibration:])

    calibration_idx = rng.permutation(np.concatenate(calibration_idx)) if calibration_idx else np.empty(0, dtype=np.int64)
    measurement_idx = rng.permutation(np.concatenate(measurement_idx)) if measurement_idx else np.empty(0, dtype=np.int64)

    if labels.any() and not labels[calibration_idx].any():
        # Nothing downstream can recover from this: F1 is 0 for every config
        # dial candidate, so _build_calibrated_filter's `f1 > best_f1` picks
        # whichever candidate it happened to try first.
        logger.warning(
            "stratified_calibration_split: calibration slice holds no duplicates, so every "
            "config dial candidate will score F1=0 and the first one tried wins by default.")
    if not labels.any():
        logger.warning(
            f"stratified_calibration_split: none of the {len(stream)} queries is within tau of "
            "the construct set, so FNR is undefined (reported as 0.0) for every filter. tau is "
            "too tight for this corpus's duplicate geometry.")

    return (stream[calibration_idx], stream[measurement_idx],
            labels[calibration_idx].tolist(), labels[measurement_idx].tolist())

