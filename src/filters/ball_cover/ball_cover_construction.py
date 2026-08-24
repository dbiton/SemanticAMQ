import numpy as np
import faiss
from faiss.contrib.inspect_tools import get_invlist


def rptree_construct_enclosing_balls(
    max_count_balls: int, vectors: np.ndarray, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    X = np.ascontiguousarray(vectors, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {X.shape}")
    n, dim = X.shape

    k = min(max_count_balls, n)
    if k <= 0:
        return np.empty((0, dim), dtype=np.float32), np.empty(0, dtype=np.float32)
    if n <= k:
        return X.copy(), np.zeros(n, dtype=np.float32)

    depth = int(np.floor(np.log2(k)))
    rng = np.random.default_rng(seed)

    # One GEMM covers every split in the tree. Per-node direction diversity comes
    # from each node drawing an index into the bank rather than a fresh vector.
    _BANK = 32
    bank = rng.standard_normal((dim, _BANK)).astype(np.float32)
    bank /= np.linalg.norm(bank, axis=0, keepdims=True)
    proj_bank = X @ bank

    rows = np.arange(n)
    seg = np.zeros(n, dtype=np.int64)
    n_seg = 1

    for _ in range(depth):
        direction = rng.integers(0, _BANK, size=n_seg)
        proj = proj_bank[rows, direction[seg]]

        # Group by segment, ordered by projection within each segment.
        order = np.lexsort((proj, seg))
        seg_sorted = seg[order]
        counts = np.bincount(seg_sorted, minlength=n_seg)
        starts = np.zeros(n_seg, dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])

        rank = rows - starts[seg_sorted]  # position within segment
        goes_right = rank >= (counts // 2)[seg_sorted]
        seg[order] = 2 * seg_sorted + goes_right
        n_seg *= 2

    order = np.argsort(seg, kind="stable")
    counts = np.bincount(seg, minlength=n_seg)
    starts = np.zeros(n_seg, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    nonempty = counts > 0
    offsets, sizes = starts[nonempty], counts[nonempty]

    Xs = X[order]  # grouped by leaf, so leaves are contiguous spans
    centers = np.add.reduceat(Xs, offsets, axis=0) / sizes[:, None]
    centers = np.ascontiguousarray(centers, dtype=np.float32)

    leaf_of = np.repeat(np.arange(sizes.size), sizes)
    diff = Xs - centers[leaf_of]
    d2 = np.einsum("ij,ij->i", diff, diff)

    radii = np.maximum.reduceat(d2, offsets)
    np.sqrt(radii, out=radii)
    return centers, radii.astype(np.float32, copy=False)


def ivf_construct_enclosing_balls(
    max_count_balls: int, vectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    X = np.ascontiguousarray(vectors, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {X.shape}")
    n, dim = X.shape

    k = min(max_count_balls, n)
    if k <= 0:
        return np.empty((0, dim), dtype=np.float32), np.empty(0, dtype=np.float32)
    if n <= k:
        return X.copy(), np.zeros(n, dtype=np.float32)
    if k == 1:
        center = X.mean(0, dtype=np.float32).reshape(1, dim)
        diff = X - center
        radius = np.sqrt(np.einsum("ij,ij->i", diff, diff).max())
        return center, np.array([radius], dtype=np.float32)

    KMEANS_ITERS = 8
    HNSW_QUANTIZER_MIN_K = 2048
    HNSW_M = 32
    HNSW_EF_CONSTRUCTION = 80
    HNSW_EF_SEARCH = 64
    KMEANS_SEED = 4263
    CHUNK = 1 << 16
    if k >= HNSW_QUANTIZER_MIN_K:
        quantizer = faiss.IndexHNSWFlat(dim, HNSW_M)
        quantizer.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        quantizer.hnsw.efSearch = HNSW_EF_SEARCH
    else:
        quantizer = faiss.IndexFlatL2(dim)

    index = faiss.IndexIVFFlat(quantizer, dim, k, faiss.METRIC_L2)
    index.cp.niter = KMEANS_ITERS
    index.cp.seed = KMEANS_SEED
    index.cp.min_points_per_centroid = 1  # silences the under-training warning
    index.train(X)
    index.add(X)

    centroids = quantizer.reconstruct_n(0, k)

    sizes = np.zeros(k, dtype=np.int64)
    id_lists = []
    for i in range(k):
        ids, _codes = get_invlist(index.invlists, i)
        sizes[i] = ids.size
        if ids.size:
            id_lists.append(ids.astype(np.int64, copy=False))

    rows = np.concatenate(id_lists)  # grouped by list, so segments are contiguous
    cell_of = np.repeat(np.arange(k, dtype=np.int64), sizes)

    # Exact distances (no norm expansion), each point against its own centroid.
    d2 = np.empty(rows.size, dtype=np.float32)
    for start in range(0, rows.size, CHUNK):
        stop = min(start + CHUNK, rows.size)
        diff = X[rows[start:stop]] - centroids[cell_of[start:stop]]
        np.einsum("ij,ij->i", diff, diff, out=d2[start:stop])

    starts = np.zeros(k, dtype=np.int64)
    np.cumsum(sizes[:-1], out=starts[1:])
    nonempty = sizes > 0

    radii = np.zeros(k, dtype=np.float32)
    radii[nonempty] = np.maximum.reduceat(d2, starts[nonempty])
    np.sqrt(radii, out=radii)

    centers = np.ascontiguousarray(centroids[nonempty])
    return centers, radii[nonempty]


def gonzales_construct_enclosing_balls(
    max_count_balls: int, vectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gonzalez farthest-first, then exact radii. O(n*k*d), single pass for
    center selection plus one exact search for radii."""
    X = np.ascontiguousarray(vectors, dtype=np.float32)
    n = X.shape[0]
    dim = X.shape[1]
    k = min(max_count_balls, n)
    if k <= 0:
        return np.empty((0, dim), dtype=np.float32), np.empty(0, dtype=np.float32)
    if n <= k:
        return X.copy(), np.zeros(n, dtype=np.float32)

    # ||x||^2, one pass, no temporaries
    sq = np.einsum("ij,ij->i", X, X)

    chosen = np.empty(k, dtype=np.int64)
    min_d2 = np.empty(n, dtype=np.float32)
    d2 = np.empty(n, dtype=np.float32)

    # d2(x, c) = ||x||^2 - 2 <x, c> + ||c||^2  -> one sgemv per center.
    # Approximate (float32 cancellation for near-duplicate x, c), but that's
    # fine here: min_d2 only drives *which* point to pick next, not any
    # radius that ends up returned -- see the exact re-search below.
    chosen[0] = 0
    np.dot(X, X[0], out=min_d2)
    min_d2 *= -2.0
    min_d2 += sq
    min_d2 += sq[0]

    for j in range(1, k):
        c = int(min_d2.argmax())
        if min_d2[c] <= 0.0:  # all points already covered exactly
            chosen = chosen[:j]
            k = j
            break
        chosen[j] = c

        np.dot(X, X[c], out=d2)
        d2 *= -2.0
        d2 += sq
        d2 += sq[c]
        np.minimum(min_d2, d2, out=min_d2)

    centers = np.ascontiguousarray(X[chosen])

    # Exact final assignment + radii via a real nearest-center search, not
    # the approximate min_d2/labels tracked during selection above: reusing
    # that running minimum directly as the squared radius let the same
    # float32 cancellation error that only needed to be "good enough" to
    # pick reasonable centers actually *understate* a returned radius by a
    # tiny margin -- rare, but a real, verified violation of the
    # zero-false-negative covering invariant every construction strategy
    # promises (caught by test_ball_cover_construction.py on clustered
    # data). The exact search costs one more O(n*k*d) pass, same order as
    # selection, and is what rptree/ivf's own radius computations already do.
    index = faiss.IndexFlatL2(dim)
    index.add(centers)
    d2_exact, labels = index.search(X, 1)
    d2_exact = np.maximum(d2_exact.ravel(), 0.0)
    labels = labels.ravel()

    # Per-cluster max via segmented reduce (np.maximum.at is much slower).
    order = np.argsort(labels, kind="stable")
    counts = np.bincount(labels, minlength=k)
    starts = np.zeros(k, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    nonempty = counts > 0

    radii = np.zeros(k, dtype=np.float32)
    radii[nonempty] = np.maximum.reduceat(d2_exact[order], starts[nonempty])
    np.sqrt(radii, out=radii)
    return centers, radii

def test():
    import matplotlib.pyplot as plt

    np.random.seed(0)
    X = np.random.randn(1000, 2).astype(np.float32)
    centers_gonzales, radii_gonzales = gonzales_construct_enclosing_balls(10, X)
    centers_ivf, radii_ivf = ivf_construct_enclosing_balls(10, X)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=1)
    for c, r in zip(centers_gonzales, radii_gonzales):
        circle = plt.Circle(c, r, color="C1", fill=False, label="Gonzalez")
        plt.gca().add_artist(circle)
    for c, r in zip(centers_ivf, radii_ivf):
        circle = plt.Circle(c, r, color="C2", fill=False, label="IVF")
        plt.gca().add_artist(circle)
    print(
        f"Gonzalez: {np.sum(radii_gonzales)} volumes, IVF: {np.sum(radii_ivf)} volume"
    )
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    test()
