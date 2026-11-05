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

Unit-normalized embeddings lie on a sphere. CapCover Filter stores a small set of **spherical caps** that together contain every indexed vector. Each cap is only a direction $u_i$ and an angle $\theta_i$:

$$
C(u_i, \theta_i) = \lbrace x \in S^{d-1} : \angle(x, u_i) \le \theta_i \rbrace.
$$

```
src/
  paper.py            entry point: runs the paper's evaluations and draws its figures
  experiments.py      evaluation definitions (generate_* submit jobs to a Runner)
  runner.py           parallel job harness (ProcessPoolExecutor) + per-job workers
  filters/            SAMQ (cap_cover/) and baseline filter implementations
  plots/               one module per evaluation's figure set
  util/                dataset path registry + embedding loaders
  vector_source.py     lazy/array-backed vector handles passed to worker processes
tests/                 unittest-style tests, one file per filter/module
datasets/              embedding pickles (not checked in, see below)
results/               generated evaluation CSVs
plots/<eval>/           generated figure PNGs
```

**False-positive bound.** Under uniform queries, the total measure of the widened caps bounds the false-positive rate:

```math
\mathrm{FPR}_\nu(\mathcal{C}, \tau) \le \min\lbrace 1, J_\tau(\mathcal{C}) \rbrace, \qquad J_\tau(\mathcal{C}) = \sum_i \sigma(\theta_i + \alpha_\tau).
```

**Space lower bound.** Take any deterministic structure that represents every set of $n$ points on the sphere in $\mathcal{B}$ bytes, with no false negatives and a uniform-query false-positive rate of at most $\varepsilon$. Then

## Running an evaluation

```
python src/paper.py              # run every evaluation, then draw the figures
python src/paper.py --only budget,streaming
python src/paper.py plot         # re-draw figures from an earlier run's results/*.csv
```

`paper.py` replaces the former `main.py`; the evaluation definitions it runs
live in [src/experiments.py](src/experiments.py). Running a generator
truncates that evaluation's CSVs in `results/` (see CLAUDE.md).

## Installation

CapCover Filter is implemented in Python with a Rust query kernel, and uses FAISS for exact search. It requires Python and the [Rust toolchain](https://rustup.rs/). The commands below install the Python dependencies and compile the Rust query kernel. Tested with Python 3.13 and Rust 1.91 on Windows.

```bash
git clone https://github.com/dbiton/SemanticAMQ.git
cd SemanticAMQ
pip install -r requirements.txt
pip install ./CapCoverRustFilter
```

## License

© 2026 Dvir David Biton, Shay Hahiashvili, Nadav Keren, Gil Einziger, Roy Friedman
