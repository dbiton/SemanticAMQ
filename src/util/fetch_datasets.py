import os
import h5py

# Increase timeout to 100 seconds (default is 10)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" # Disable fast transfer if enabled, sometimes causes issues
os.environ["HF_HUB_ETAG_TIMEOUT"] = "100"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "100"

import numpy as np
from typing import *
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from datasets.combine import concatenate_datasets
from scipy.cluster.hierarchy import DisjointSet
from sklearn.preprocessing import normalize

# from util.dataset_paths import dataset_filenames  # load_embeds/load_meta both index this

embeds_dir = "datasets"
MAX_STREAM_LENGTH = 500_000
models: Dict[str, SentenceTransformer] = {}

def load_embeds(dataset_name: str, N: int):
    path = dataset_filenames[dataset_name]
    with h5py.File(path, "r") as f:
        embeds = f["normalized_embeds"][:N]
        text_bytes = f["text"][:N]
        embeds_texts = [t.decode("utf-8") for t in text_bytes]
    embeds = np.array(embeds, dtype=np.float32)
    return embeds, embeds_texts

def embed_strings(strings: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    if model_name not in models:
        models[model_name] = SentenceTransformer(model_name)
        print(f"Loaded model {model_name} into device {models[model_name].device}")
    model = models[model_name]
    embs = model.encode(strings, convert_to_numpy=True, show_progress_bar=True)
    return embs

@contextmanager
def writer(path: str):
    if os.path.exists(path):
        f = None
    else:
        f = NamedTemporaryFile(mode="w+b", delete=False)
        tmp_path = f.name
    try:
        yield f
        if f is not None:
            f.close()  # Ensure closed before rename
            os.rename(tmp_path, path)
    except:
        if f is None:
            print(f'Skipping "{path}" because it already exists')
        else:
            f.close()
            os.remove(tmp_path)
            raise

def _write_meta(hf, meta: Dict[str, List[Any]], n: int) -> None:
    """Persist each metadata column alongside the embeddings, truncated to the
    same n rows so column i always describes embedding i.

    Without this the metadata was asserted and then dropped, which silently
    cost the two columns that carry ground truth rather than decoration:
    build_quora's `set_id` (the transitive closure of the duplicate-label
    graph -- the only way to recover Quora's labelled paraphrase clusters,
    since they are NOT recoverable from the embeddings) and build_ccnews's
    `domain` (what shows a duplicate cluster spans multiple outlets, i.e. that
    it is syndication rather than one site reposting itself).

    Integer columns are stored as int64, everything else as UTF-8 strings;
    None becomes "" (several builders pad unused columns with None).
    """
    dt = h5py.string_dtype(encoding="utf-8")
    for key, values in meta.items():
        col = list(values)[:n]
        if col and all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in col):
            hf.create_dataset(f"meta/{key}", data=np.asarray(col, dtype=np.int64))
        else:
            col = ["" if v is None else str(v).replace("\x00", "") for v in col]
            hf.create_dataset(f"meta/{key}", data=col, dtype=dt)


def pack_and_dump(path: str, texts: List[str], meta: Dict[str, List[Any]], model_name: str = "all-MiniLM-L6-v2"):
    assert all(len(v) == len(texts) for v in meta.values()), "Metadata length mismatch"
    with writer(path) as f:
        if f is None:
            print(f'Skipping "{path}" because it already exists')
            return
        # Everything below is truncated to the number of vectors actually
        # embedded. "text" used to be written at full length while "embeds"
        # stopped at MAX_STREAM_LENGTH, so the two had different row counts on
        # any corpus larger than the cap -- harmless only because both start at
        # row 0. Meta columns must line up with the embeddings to be usable as
        # labels at all, so all three are cut to the same n here.
        texts = texts[:MAX_STREAM_LENGTH]
        embeds = embed_strings(texts, model_name=model_name)
        normalized_embeds = normalize(embeds)
        n = len(embeds)
        with h5py.File(f, "w") as hf:
            hf.create_dataset("embeds", data=embeds)
            hf.create_dataset("normalized_embeds", data=normalized_embeds)
            dt = h5py.string_dtype(encoding='utf-8')
            texts = [t.replace('\x00', '') for t in texts[:n]]
            hf.create_dataset("text", data=texts, dtype=dt)
            _write_meta(hf, meta, n)
    print(f"Saved HDF5 to {path}")


def load_meta(dataset_name: str, N: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Metadata columns written by _write_meta, as {column: array of N rows}.

    Returns {} for the datasets built before metadata was persisted -- their
    .pkl predates _write_meta and has no "meta" group, so a caller that needs
    e.g. Quora's set_id must delete that file and rebuild.
    """
    path = dataset_filenames[dataset_name]
    out: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        if "meta" not in f:
            return out
        for key in f["meta"]:
            col = f["meta"][key][:N] if N is not None else f["meta"][key][:]
            if col.dtype.kind in ("S", "O"):
                col = np.array([v.decode("utf-8") if isinstance(v, bytes) else v for v in col], dtype=object)
            out[key] = col
    return out

def build_eli5() -> Tuple[List[str], Dict[str, List[Any]]]:
    ds = load_dataset("sentence-transformers/eli5", trust_remote_code=True)
    texts: List[str] = ds["train"]["question"]
    meta = {}
    return texts, meta

def build_wildchat() -> Tuple[List[str], Dict[str, List[Any]]]:
    ds = load_dataset("allenai/WildChat-1M")
    texts, session_id, turn_id = [], [], []
    sid = 0
    for ex in ds["train"]:
        if ex.get("language") != "English":
            continue
        # Keep only user turns; preserve order
        user_turns = []
        for t in ex.get("conversation", []):
            role = t.get("role") or t.get("author_role")  # robustness
            if role in ("user", "User"):
                content = t.get("content") or t.get("text")
                if content:
                    user_turns.append(content)
        if not user_turns:
            continue
        for k, ut in enumerate(user_turns):
            texts.append(ut)
            session_id.append(f"wildchat_{sid}")
            turn_id.append(k)
        sid += 1
    meta = {
        "session_id": session_id,
        "turn_id": turn_id,
    }
    return texts, meta

def build_qrecc() -> Tuple[List[str], Dict[str, List[Any]]]:
    ds = load_dataset("svakulenk0/qrecc")
    texts, session_id, turn_id = [], [], []
    # Fields: 'conversation_id' (or 'qid_conv'), 'turn_id' (or 'turn'), 'question'
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        for ex in ds[split]:
            q = ex.get("question") or ex.get("rewritten_question") or ex.get("raw_question")
            if not q:
                continue
            cid = ex.get("conversation_id") or ex.get("qid_conv") or ex.get("conversation_no") or "unk"
            tid = ex.get("turn_id") or ex.get("turn") or 0
            texts.append(q)
            session_id.append(f"qrecc_{cid}")
            turn_id.append(int(tid))
    # Sort by (session, turn) to preserve order
    order = sorted(range(len(texts)), key=lambda i: (session_id[i], turn_id[i]))
    texts = [texts[i] for i in order]
    session_id = [session_id[i] for i in order]
    turn_id = [turn_id[i] for i in order]
    meta = {
        "dataset": ["QReCC"] * len(texts),
        "session_id": session_id,
        "turn_id": turn_id,
        "language": ["en"] * len(texts),
        "timestamp": [None] * len(texts),
        "pair_id": [None] * len(texts),
    }
    return texts, meta

def build_stackoverflow() -> Tuple[List[str], Dict[str, List[Any]]]:
    ds = load_dataset("pacovaldez/stackoverflow-questions")
    # Can also get the questions themselves
    texts = concatenate_datasets([ds[s] for s in ('train','validation','test')])['title']
    meta = {}
    return texts, meta

def build_quora() -> Tuple[List[str], Dict[str, List[Any]]]:
    ds = load_dataset("quora", trust_remote_code=True)
    # structure: each row has 'questions': [{'text': q1}, {'text': q2}], possibly 'id'
    texts, question_id = [], []
    union_find = DisjointSet()
    for ex in ds["train"]:
        q1, q2 = ex['questions']['text']
        q1_id, q2_id = ex['questions']['id']
        union_find.add(q1_id)
        union_find.add(q2_id)
        if ex['is_duplicate']:
            union_find.merge(q1_id, q2_id)
        texts += [q1, q2]
        question_id += [q1_id, q2_id]
    set_id = [union_find[qid] for qid in question_id]
    meta = {
        "set_id": set_id,
        "question_id": question_id
    }
    return texts, meta

def build_mmlu() -> Tuple[List[str], Dict[str, List[Any]]]:
    """MMLU ('all' config: every subject, and the test/validation/dev/
    auxiliary_train splits concatenated).

    Embeds the question TOGETHER WITH its answer choices, rather than the
    question alone. That is the faithful representation of an MMLU item: 31
    questions in a 9k sample recur with DIFFERENT choice sets (e.g. "The
    terrestrial planet cores contain mostly metal because"), and with the
    choices dropped those distinct items were indistinguishable.

    It does NOT, however, fix MMLU's inflated duplicate rate, and it is worth
    recording why so nobody re-attempts this. Measured on 9k items:

        rows repeating a question exactly            5.5%
        rows that are a fully repeated item          5.2%
        pairs within embedding distance <= 0.01     29.9%  (question only)
                                                    29.2%  (question+choices)

    String-level duplication is ~5%, but distance-based duplication is ~30%,
    and adding the choices barely moves it. The gap is not an extraction bug:
    it is all-MiniLM-L6-v2 failing to resolve short, formulaic multiple-choice
    text, mapping textually distinct items to within 0.01 of each other. The
    choices are formulaic too, so they add little separating signal.

    Consequence: MMLU's nonzero redundancy at small tau in system_tau.csv
    (redundancy_rate 0.0120 at tau=0.01, against 0.0000 for ELI5/HotpotQA/
    MsMarco) measures encoder resolution, not duplicate structure -- and this
    builder cannot fix that. Treating MMLU as a duplicate corpus needs either a
    higher-resolution encoder or ground truth defined by string/label identity
    instead of distance.

    Choices are joined with " | " so the option boundaries survive
    tokenization. Note MiniLM truncates at 256 word-pieces, so for the few
    long-passage items in auxiliary_train the trailing choices can still be
    cut -- those items remain under-disambiguated, just far fewer of them.

    Keeps subject/answer/split in metadata: `subject` is a ready-made topical
    grouping (57 of them), which is a natural source of hard negatives -- items
    that are lexically close but semantically distinct, MMLU's real strength
    (template variants such as "the degree of Q(sqrt(2) + sqrt(3))" vs
    "Q(sqrt(2), sqrt(3))" sit ~0.2 apart yet have different answers).
    """
    print("Loading MMLU (config: 'all')...")
    ds = load_dataset("cais/mmlu", "all")

    texts, subject, answer, split_name = [], [], [], []
    # Splits: 'test', 'validation', 'dev', 'auxiliary_train'
    for split in ds:
        feats = ds[split].features
        if "question" not in feats:
            continue
        questions = ds[split]["question"]
        n = len(questions)
        choices = ds[split]["choices"] if "choices" in feats else [None] * n
        subjects = ds[split]["subject"] if "subject" in feats else [""] * n
        answers = ds[split]["answer"] if "answer" in feats else [-1] * n
        for q, ch, sub, ans in zip(questions, choices, subjects, answers):
            options = " | ".join(str(c) for c in ch) if ch else ""
            texts.append(f"{q} {options}".strip() if options else q)
            subject.append(sub or "")
            answer.append(int(ans) if isinstance(ans, (int, np.integer)) else -1)
            split_name.append(split)

    meta = {
        "subject": subject,
        "answer": answer,
        "split": split_name,
    }
    return texts, meta

def build_hotpotqa() -> Tuple[List[str], Dict[str, List[Any]]]:
    # HotpotQA (Multi-hop reasoning). The unnamespaced "hotpot_qa" repo id was
    # script-based and no longer loads on datasets>=4.0 ("dataset scripts are
    # no longer supported"); "hotpotqa/hotpot_qa" is the maintainers' own
    # Parquet-converted mirror, same "distractor" config and "question" field.
    print("Loading HotpotQA...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    texts = []
    # Splits: 'train', 'validation'
    for split in ds:
        texts.extend(ds[split]["question"])
        
    meta = {}
    return texts, meta

def build_triviaqa() -> Tuple[List[str], Dict[str, List[Any]]]:
    # TriviaQA
    # 'rc' (Reading Comprehension) is the standard configuration
    print("Loading TriviaQA (config: 'rc')...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc")
    
    texts = []
    # Splits: 'train', 'validation', 'test'
    for split in ds:
        texts.extend(ds[split]["question"])
        
    meta = {}
    return texts, meta

def build_nq() -> Tuple[List[str], Dict[str, List[Any]]]:
    texts = []
    ds = load_dataset("nq_open")
    for split in ("train", "validation"):
        texts.extend([q for q in ds[split]["question"] if q])
    meta = {}
    return texts, meta

def build_msmarco() -> Tuple[List[str], Dict[str, List[Any]]]:
    texts = []
    ds = load_dataset("ms_marco", "v2.1")
    for split in ("train", "validation", "test"):
        texts.extend([q for q in ds[split]["query"] if q])
    meta = {}
    return texts, meta

def build_pes2o() -> Tuple[List[str], Dict[str, List[Any]]]:
    # allenai/peS2o itself is still script-based (a custom loading script),
    # which datasets>=4.0 refuses to run at all ("dataset scripts are no
    # longer supported") -- trust_remote_code=True no longer has any effect,
    # since the script-execution mechanism itself was removed from the
    # library. common-pile/peS2o is a Parquet-native, openly-licensed subset
    # derived from the same underlying peS2o/S2ORC corpus (~6.3M docs, not a
    # byte-identical mirror of the original ~39M-doc allenai/peS2o -- see its
    # dataset card), with the same "text" field. Use streaming to avoid
    # gigabytes of downloads; we only pull the first MAX_STREAM_LENGTH docs.
    print("Loading peS2o (streaming first 1M documents)...")
    ds = load_dataset("common-pile/peS2o", split="train", streaming=True)
    
    texts = []
    for i, ex in enumerate(ds):
        if i >= MAX_STREAM_LENGTH:
            break
        if ex.get("text"):
            texts.append(ex["text"])
            
    meta = {}
    return texts, meta

def build_ccnews() -> Tuple[List[str], Dict[str, List[Any]]]:
    """CC-News: news articles from CommonCrawl News (2017-2019), via the
    Parquet-native vblagoje/cc_news mirror (no loading script, so it still
    works on datasets>=4.0 -- see build_pes2o for why that matters).

    Chosen over the other news corpora specifically for *syndication*: the
    same wire story (AP/Reuters) republished across outlets with a rewritten
    headline and lightly edited lede. That produces high-multiplicity clusters
    of near-duplicates that are NOT byte-identical, which is the one property
    WildChat cannot supply -- its duplicate clusters are mostly exact reposts,
    so a hash set answers them and a geometric filter earns nothing.

    Deliberately NOT Multi-News or NewSHead: those cluster *different*
    articles covering the same event, which are topical siblings, not
    near-duplicates. Measured L2 distance between topical siblings is ~0.79 on
    this embedding model, far too large for a ball cover to exploit (a ball
    that wide covers the whole corpus). Syndicated copies sit at ~0.1, which
    is the regime the filter works in.

    The article body is embedded rather than the title: outlets rewrite
    headlines more often than they rewrite wire copy, and MiniLM's 256-token
    window truncates to roughly the lede -- the most-copied part of the story.

    MIN_CHARS drops stubs (paywall notices, "enable JavaScript", truncated
    feeds). Those are byte-identical boilerplate repeated across thousands of
    pages, and they would manufacture exactly the giant *exact*-duplicate
    clusters this dataset was picked to avoid.
    """
    MIN_CHARS = 400

    print(f"Loading CC-News (streaming first {MAX_STREAM_LENGTH} articles)...")
    ds = load_dataset("vblagoje/cc_news", split="train", streaming=True)
    # The stream arrives in per-domain runs (measured: the first 60k records
    # cover only 752 domains, in blocks). A contiguous prefix would therefore
    # over-sample a handful of outlets -- and syndication is by definition a
    # CROSS-outlet phenomenon, so that prefix would under-represent the very
    # clusters this dataset was added to provide. Shuffling the shard order
    # and a read buffer spreads the sample across publishers.
    ds = ds.shuffle(seed=42, buffer_size=50_000)

    texts, title, domain, date, url = [], [], [], [], []
    seen = 0
    for ex in ds:
        body = (ex.get("text") or "").strip()
        if len(body) < MIN_CHARS:
            continue
        texts.append(body)
        title.append(ex.get("title") or "")
        # Publisher, kept so a duplicate cluster can be shown to span multiple
        # outlets -- the evidence that a cluster is syndication rather than one
        # site reposting itself.
        domain.append(ex.get("domain") or "")
        date.append(ex.get("date") or "")
        url.append(ex.get("url") or "")
        seen += 1
        if seen >= MAX_STREAM_LENGTH:
            break

    meta = {
        "title": title,
        "domain": domain,
        "date": date,
        "url": url,
    }
    return texts, meta


def build_wikianswers() -> Tuple[List[str], Dict[str, List[Any]]]:
    """WikiAnswers paraphrase clusters, via the Parquet-native
    embedding-data/WikiAnswers mirror. Each row is a `set` of user-submitted
    questions that WikiAnswers editors marked as asking the same thing.

    Added for the one property neither WildChat nor CC-News supplies:
    duplicates that are REPHRASINGS rather than reposts. Measured on a
    40k-question sample of this corpus:

        exact duplicates (d <= 0.01)       0.2%
        rephrasings      (0.01 < d <= 0.4) 53.2%
        intra-cluster NN distance          median 0.376 (p25 0.235, p75 0.534)
        ambient pairwise distance          median 1.354
        cluster size                       median 27, max 51

    Compare WildChat and CC-News, where ~84% of duplicates are byte-identical
    and a hash set answers them. Here a hash set catches essentially nothing,
    which is what makes this corpus the one that can justify a *geometric*
    filter -- while the clusters stay far tighter than topical siblings
    (~0.79), so the cover still has room to work.

    `set_id` is written to metadata, so ground truth comes from the editors'
    labels instead of from a distance threshold. That matters beyond
    convenience: everywhere else "duplicate" is defined as "within tau", the
    same quantity the filter thresholds, which makes FPR partly circular.
    Labels break that circularity.

    Singleton sets are dropped -- with no partner they can never be a true
    positive, and would only pad the corpus with guaranteed negatives.
    """
    print(f"Loading WikiAnswers (streaming up to {MAX_STREAM_LENGTH} questions)...")
    ds = load_dataset("embedding-data/WikiAnswers", split="train", streaming=True)

    texts, set_id = [], []
    for i, ex in enumerate(ds):
        members = [t for t in (ex.get("set") or []) if t]
        if len(members) < 2:
            continue
        for t in members:
            texts.append(t)
            set_id.append(i)
        if len(texts) >= MAX_STREAM_LENGTH:
            break

    meta = {"set_id": set_id}
    return texts, meta


if __name__ == "__main__":
    os.makedirs(embeds_dir, exist_ok=True)

    try:
        print("generating StackOverflow...")
        texts, meta = build_stackoverflow()
        pack_and_dump(os.path.join(embeds_dir, "embeds_stackoverflow.pkl"), texts, meta)
    except Exception as e:
        print(f"Failed to generate StackOverflow: {e}")

    try:
        print("generating Quora Question Pairs...")
        texts, meta = build_quora()
        pack_and_dump(os.path.join(embeds_dir, "embeds_quora_qp.pkl"), texts, meta)
    except Exception as e:
        print(f"Failed to generate Quora Question Pairs: {e}")

    try:
        print("generating WildChat...")
        texts, meta = build_wildchat()
        pack_and_dump(os.path.join(embeds_dir, "embeds_wildchat.pkl"), texts, meta)
    except Exception as e:
        print(f"Failed to generate WildChat: {e}")

    try:
        print("generating TriviaQA...")
        texts, meta = build_triviaqa()
        pack_and_dump(os.path.join(embeds_dir, "embeds_triviaqa.pkl"), texts, meta)
    except Exception as e:
        print(f"Failed to generate TriviaQA: {e}")

    try:
        print("generating HotpotQA...")
        texts, meta = build_hotpotqa()
        pack_and_dump(os.path.join(embeds_dir, "embeds_hotpotqa.pkl"), texts, meta)
    except Exception as e:
        print(f"Failed to generate HotpotQA: {e}")

    try:
        print("generating MMLU...")
        texts, meta = build_mmlu()
        pack_and_dump(os.path.join(embeds_dir, "embeds_mmlu.pkl"), texts, meta)
    except Exception as e:
        print(f"Failed to generate MMLU: {e}")

    try:
        print("generating ELI5...")
        texts, meta = build_eli5()
        pack_and_dump(os.path.join(embeds_dir, "embeds_eli5.pkl"), texts, meta)
    except Exception as e:
        print(f"Failed to generate ELI5: {e}")

    try:
        print("generating Natural Questions...")
        texts, meta = build_nq()
        pack_and_dump(os.path.join(embeds_dir, "embeds_nq.pkl"), texts, meta)
    except Exception as e:
        print(f"Failed to generate Natural Questions: {e}")

    try:
        print("generating MS MARCO...")
        texts, meta = build_msmarco()
        pack_and_dump(os.path.join(embeds_dir, "embeds_msmarco.pkl"), texts, meta)
    except Exception as e:
        print(f"Failed to generate MS MARCO: {e}")

    try:
        print("generating peS2o...")
        texts, meta = build_pes2o()
        pack_and_dump(os.path.join(embeds_dir, "embeds_pes2o.pkl"), texts, meta)
    except Exception as e:
        print(f"Failed to generate peS2o: {e}")

    try:
        print("generating CC-News...")
        texts, meta = build_ccnews()
        pack_and_dump(os.path.join(embeds_dir, "embeds_ccnews.pkl"), texts, meta)
    except Exception as e:
        print(f"Failed to generate CC-News: {e}")

    try:
        print("generating WikiAnswers...")
        texts, meta = build_wikianswers()
        pack_and_dump(os.path.join(embeds_dir, "embeds_wikianswers.pkl"), texts, meta)
    except Exception as e:
        print(f"Failed to generate WikiAnswers: {e}")

    print("Done.")