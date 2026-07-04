"""
Lightweight dataset path registry.

Kept free of heavy imports (no torch / sentence_transformers / datasets) on
purpose: worker processes and VectorSource only need to map a dataset name to
its on-disk HDF5 file. Pulling in the ML stack here would add ~400 MB of RSS to
every spawned worker for nothing.
"""

# HDF5 dataset key that the normalized vectors are stored under.
HDF5_DATASET_KEY = "normalized_embeds"

dataset_filenames = {
    "MsMarco": "./datasets/embeds_msmarco.pkl",
    "WildChat": "./datasets/embeds_wildchat.pkl",
    "ELI5": "./datasets/embeds_eli5.pkl",
    "NaturalQuestions": "./datasets/embeds_nq.pkl",
    "StackOverflow": "./datasets/embeds_stackoverflow.pkl",
    "Quora": "./datasets/embeds_quora_qp.pkl",
    "MMLU": "./datasets/embeds_mmlu.pkl",
    "TriviaQA": "./datasets/embeds_triviaqa.pkl",
    "HotPotQA": "./datasets/embeds_hotpotqa.pkl",
    "peS2o": "./datasets/embeds_pes2o.pkl",
}
