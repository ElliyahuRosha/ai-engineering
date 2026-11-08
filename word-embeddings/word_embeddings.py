"""Word embeddings from scratch: corpus, vocabulary and training pairs."""

import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

# ASCII letters only, with an optional internal apostrophe ("don't").
# Non-English words, numbers and punctuation are dropped on purpose.
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

# Single id reserved for words below the frequency threshold and unseen words.
UNK = "<unk>"


def load_corpus(path: Path) -> str:
    """Read the corpus file as one lowercased string."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"corpus is empty: {path}")
    return text.lower()


def tokenize(text: str) -> list[str]:
    """Split text into word tokens."""
    return TOKEN_RE.findall(text)


def build_vocab(tokens: list[str], min_count: int = 5) -> dict[str, int]:
    """Map frequent words to consecutive ids. Id 0 is always UNK."""
    counts = Counter(tokens)
    kept = [pair for pair in counts.items() if pair[1] >= min_count]
    kept.sort(key=lambda pair: (-pair[1], pair[0]))  # stable, reproducible ids
    vocab = {UNK: 0}
    for word, _ in kept:
        vocab[word] = len(vocab)
    return vocab


def encode(tokens: list[str], vocab: dict[str, int]) -> np.ndarray:
    """Replace tokens with ids, sending rare and unseen words to UNK.

    np.fromiter fills the array directly. A plain list comprehension would need
    8 bytes per token of pointers - 3.7 GB on a 470M token corpus, for nothing.
    """
    return np.fromiter(
        (vocab.get(token, 0) for token in tokens), dtype=np.int32, count=len(tokens)
    )


def cooccurrence(ids: list[int], vocab_size: int, window: int = 5) -> sparse.csr_matrix:
    """Count symmetric word-context pairs inside a fixed window."""
    ids = np.asarray(ids, dtype=np.int32)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for offset in range(1, window + 1):
        # one diagonal of the matrix per offset, in both directions
        rows.append(ids[:-offset])
        cols.append(ids[offset:])
        rows.append(ids[offset:])
        cols.append(ids[:-offset])
    counts = sparse.coo_matrix(
        (np.ones(sum(chunk.size for chunk in rows), dtype=np.int32),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(vocab_size, vocab_size),
    )
    return counts.tocsr()


def ppmi(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    """Positive PMI: keep only pairs that are more frequent than chance.

    PMI(w, c) = log( count(w, c) * total / (count(w) * count(c)) )
    """
    total = float(counts.sum())
    word_totals = np.asarray(counts.sum(axis=1)).ravel()
    context_totals = np.asarray(counts.sum(axis=0)).ravel()

    pairs = counts.tocoo()
    observed = pairs.data * total
    expected = word_totals[pairs.row] * context_totals[pairs.col]
    pmi = np.log(observed / expected)
    pmi = np.maximum(pmi, 0)  # negative PMI means "less often than chance"

    return sparse.coo_matrix(
        (pmi, (pairs.row, pairs.col)), shape=counts.shape
    ).tocsr()


def window_pairs(ids: np.ndarray, position: int, window: int = 5) -> list[int]:
    """Return the context ids around one position, inside the window."""
    start = max(position - window, 0)
    end = min(position + window + 1, len(ids))
    return [int(ids[i]) for i in range(start, end) if i != position]


def sample_pairs(
    ids: np.ndarray, batch_size: int, window: int = 5, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Draw random (center, context) pairs instead of materialising all of them.

    20M tokens x 10 neighbours = 200M pairs; sampling keeps memory flat.
    Pairs touching UNK (id 0) are rejected and redrawn: it is a bucket, not a word.
    """
    rng = rng or np.random.default_rng()
    centers = np.empty(batch_size, dtype=np.int32)
    contexts = np.empty(batch_size, dtype=np.int32)

    filled = 0
    while filled < batch_size:
        missing = batch_size - filled
        positions = rng.integers(window, len(ids) - window, size=missing)
        steps = rng.integers(1, window + 1, size=missing) * rng.choice([-1, 1], size=missing)
        context_positions = positions + steps
        keep = (ids[positions] != 0) & (ids[context_positions] != 0)
        taken = int(keep.sum())
        centers[filled:filled + taken] = ids[positions[keep]]
        contexts[filled:filled + taken] = ids[context_positions[keep]]
        filled += taken

    return centers, contexts


def subsample_ids(
    ids: np.ndarray, threshold: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Drop frequent words, as word2vec does, so "the" stops eating the window.

    Each token survives with probability sqrt(threshold / frequency), capped at 1.
    `threshold` is a frequency: 1e-4 means "a word seen in more than one token in
    ten thousand is cut down". Words below it are always kept.
    """
    rng = rng or np.random.default_rng()
    counts = np.bincount(ids, minlength=int(ids.max()) + 1)
    frequency = counts / counts.sum()
    keep = np.ones_like(frequency)
    common = frequency > threshold
    keep[common] = np.sqrt(threshold / frequency[common])
    keep[0] = 1.0  # UNK is a bucket, not a word: never drop it
    return ids[rng.random(ids.size) < keep[ids]]


def negative_distribution(
    ids: np.ndarray, vocab_size: int, power: float = 0.75
) -> np.ndarray:
    """Unigram counts raised to `power`, normalised.

    power=1.0 samples words exactly by frequency; 0.75 flattens the head so
    common words still dominate but rare words are not ignored.
    """
    counts = np.bincount(ids, minlength=vocab_size).astype(np.float64)
    weights = counts**power
    weights[0] = 0.0  # UNK is a bucket, not a word: never draw it as a negative
    return weights / weights.sum()


def sample_negatives(
    batch_size: int,
    num_negatives: int,
    probabilities: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw random word ids to act as negative examples."""
    rng = rng or np.random.default_rng()
    cumulative = np.cumsum(probabilities)
    cumulative /= cumulative[-1]   # guard against float drift below 1.0
    cumulative[-1] = 1.0
    draws = rng.random((batch_size, num_negatives))
    return np.searchsorted(cumulative, draws).astype(np.int32)


if __name__ == "__main__":
    tokens = tokenize(load_corpus(Path("data/tinystories_20m.txt")))
    vocab = build_vocab(tokens)
    ids = np.asarray(encode(tokens, vocab), dtype=np.int32)
    words = list(vocab)
    print(f"tokens: {len(tokens):,}   vocab: {len(vocab):,}")

    start = 5_000
    center = start + 12
    print("text   :", " ".join(tokens[start:start + 24]))
    print("center :", words[ids[center]])
    print("neigh  :", [words[c] for c in window_pairs(ids, center, window=3)])

    rng = np.random.default_rng(0)
    centers, contexts = sample_pairs(ids, 8, window=5, rng=rng)
    print("sampled:", [(words[c], words[x]) for c, x in zip(centers, contexts)])

    probs = negative_distribution(ids, len(vocab))
    flat = negative_distribution(ids, len(vocab), power=1.0)
    for word in ["the", "cat", "mountain"]:
        print(f"P({word:8}) = {probs[vocab[word]]:.5f}  vs unigram {flat[vocab[word]]:.5f}")

    centers, contexts = sample_pairs(ids, 1, window=5, rng=rng)
    negatives = sample_negatives(1, 5, probs, rng)
    print(f"positive : ({words[centers[0]]}, {words[contexts[0]]})")
    print(f"negatives: {[words[i] for i in negatives[0]]}")
