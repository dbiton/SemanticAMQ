<p align="center">
  <img src="assets/logo.svg" alt="CapCover Filter" width="460">
</p>

<h3 align="center">Semantic Approximate Membership Queries over Embeddings</h3>

<p align="center">
  Know when a vector search can be skipped, with zero false negatives.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Rust-B7410E?style=flat-square&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/FAISS-0467DF?style=flat-square&logo=meta&logoColor=white" alt="FAISS">
</p>

<p align="center">
  <a href="#the-problem">The problem</a> &nbsp;·&nbsp;
  <a href="#how-it-works">How it works</a> &nbsp;·&nbsp;
  <a href="#building-a-cover">Building a cover</a> &nbsp;·&nbsp;
  <a href="#results">Results</a> &nbsp;·&nbsp;
  <a href="#when-to-use-it">When to use it</a> &nbsp;·&nbsp;
  <a href="#theory">Theory</a> &nbsp;·&nbsp;
  <a href="#installation">Installation</a>
</p>

<br>

<table align="center">
  <tr>
    <td align="center" width="200"><h3>0</h3>false negatives, by construction</td>
    <td align="center" width="200"><h3>2%</h3>of the raw embedding memory</td>
    <td align="center" width="200"><h3>20–100%</h3>of index accesses avoided</td>
    <td align="center" width="200"><h3>1</h3>filter for any query threshold</td>
  </tr>
</table>

<p align="center"><sub>Memory and savings measured at cosine similarity 0.95 on five real-world corpora; savings depend on the corpus.</sub></p>

## The problem

Semantic caches, retrieval pipelines and de-duplication systems often need a yes-or-no answer: **is any stored embedding similar enough to this query?** Answering that normally takes a vector search, even when the answer is no.

We call this a **Semantic Approximate Membership Query (SAMQ)**, the semantic counterpart of what a Bloom filter does for exact keys. CapCover Filter is a compact structure that sits in front of the index and answers it first. For an indexed set $X$ of unit vectors, a query $q$ and a cosine-distance threshold $\tau$, the exact predicate is

$$
M_X(q, \tau) = \mathbf{1}\left[\min_{x \in X} \bigl(1 - \langle q, x \rangle\bigr) \le \tau\right].
$$

The filter says YES whenever $M_X(q, \tau) = 1$, and may also say YES for some other queries. A NO is always correct, so a rejected query can safely skip the index.

<p align="center">
  <img src="assets/query-flow.svg" alt="A query goes to the CapCover Filter. On NO the vector search is skipped. On YES the vector index verifies the answer." width="780">
</p>

## How it works

SAMQ (Cap-Cover) builds a cover of spherical caps over the construct set (direction
and angle per cap, quantizable to shrink memory -- see `Quantize` in
[cap_cover_filter.py](src/filters/cap_cover/cap_cover_filter.py)),
then answers a query by checking whether it falls within any cap's
angle (widened by the query threshold). It is compared at equal memory against the
baselines in [src/filters/](src/filters/): DSH, HNSW, IVF-PQ, a quantized flat
index, VA+-file, an MLP, and two zero-false-negative LSH schemes.

$$
C(u_i, \theta_i) = \lbrace x \in S^{d-1} : \angle(x, u_i) \le \theta_i \rbrace.
$$

```
src/
  paper.py            the entry point: runs the paper's evaluations, draws its figures and tables
  experiments.py      evaluation definitions (generate_* submit jobs to a Runner)
  workloads.py        positional folds over a corpus file, exact ground truth
  runner.py           parallel job harness (ProcessPoolExecutor) + per-job workers
  filters/            Cap-Cover (cap_cover/) and the baselines
  plots/              plot_paper.py (figures), tables_paper.py, table_quantization.py (LaTeX tables)
  util/               dataset registry (import-light), dataset builders, space bound
  vector_source.py    lazy/array-backed vector handles passed to worker processes
CapCoverRustFilter/   Rust crate behind CapCoverFilter.query()
tests/                unittest-style tests, one file per filter/module
datasets/             embedding files (HDF5 despite .pkl; not checked in)
results/              generated CSVs (not checked in)
figures/              generated figures and tables
```

**False-positive bound.** Under uniform queries, the total measure of the widened caps bounds the false-positive rate:

Python 3.12 with faiss-cpu 1.15.1 is the tested combination.

```
pip install -r requirements.txt            # evaluation and plotting
pip install maturin && cd CapCoverRustFilter && maturin develop --release   # the Rust scan
pip install -r requirements-data.txt       # only to build datasets (sentence_transformers)
python src/util/fetch_datasets.py          # the paper's five corpora -> datasets/
```

The evaluations read `datasets/embeds_<corpus>.pkl` for WildChat, Quora, WikiAnswers,
ChatGPT-Paraphrases and CC-News (see `dataset_filenames` in
[src/util/dataset_paths.py](src/util/dataset_paths.py)); each needs 500,000 rows for the
five-fold 100,000-vector runs.

## Running the evaluations

Run from the repository root.

```
python src/paper.py --list                       # evaluations, their scale and outputs
python src/paper.py                              # everything, then figures and tables
python src/paper.py --only budget,streaming      # a subset
python src/paper.py --missing                    # only evaluations without complete CSVs
python src/paper.py plot [figure ...]            # re-draw from existing CSVs; needs no datasets
python src/paper.py --num-vectors 10000 --folds 1 --results /tmp/small    # a reduced trial run
```

Each evaluation runs at the scale paper.tex states for it (`paper.EVALUATIONS`): the sweeps and
end-to-end throughput at 100,000 vectors x 5 folds, the quantization table, insertion, equal-memory
and ablation runs at 10,000 vectors x 1 fold, latency on WildChat (20,000 indexed) and WikiAnswers
(8,000). **Running an evaluation rewrites its CSVs** in `--results` (default `results/`); collecting
jobs does not.

Hardware: the full run was made on a 32-core Threadripper with 896 GB RAM (30 parallel workers,
about 10 GB per job). The timed evaluations (end-to-end on 8 pinned cores; latency and timing on one)
run after the parallel batch and need an otherwise idle machine; timings from the parallel batch are
not timing claims.

## Tests

```
pip install -r requirements-dev.txt
python -m pytest tests
```

Tests that need the Rust extension's newer kernels or AVX-512 skip when they are unavailable; rebuild
the extension (above) so the Python/Rust parity tests run.

## Outputs

CSVs in `results/` (one file per evaluation, or per corpus); figures and LaTeX tables in `figures/`
with a `plot-manifest.json` recording each figure's input files, row counts and hashes.

## Release tree

`.gitattributes` marks scratch directories (`suggestions/`, `bench_construct/`, `tasks/`) `export-ignore`,
so `git archive HEAD` produces the publication tree without them.
