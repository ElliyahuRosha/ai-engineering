"""Score a trained run: neighbours, analogies, and the floor they have to beat.

Reads `embeddings.npy` and `vocab.txt` from a run directory, writes `metrics.json`
next to them, and prints a one-line summary.
"""

import argparse
import json
from pathlib import Path

import numpy as np

GROUPS = {
    "royalty": ["king", "queen", "prince", "princess"],
    "family": ["mom", "dad", "brother", "sister"],
    "animals": ["dog", "cat", "bird", "fish", "bear", "rabbit"],
    "places": ["castle", "forest", "garden", "house", "school"],
    "feelings": ["happy", "sad", "scared", "angry"],
    "size": ["big", "small"],
}

# a -> b is the relation; apply it to c and expect `expected`.
# Written from the language, not from what the model happens to get right.
ANALOGIES = [
    ("king", "queen", "prince", "princess"),
    ("king", "queen", "man", "woman"),
    ("boy", "girl", "man", "woman"),
    ("boy", "girl", "brother", "sister"),
    ("mom", "dad", "mother", "father"),
    ("mom", "dad", "sister", "brother"),
    ("son", "daughter", "brother", "sister"),
    ("cat", "cats", "dog", "dogs"),
    ("bird", "birds", "fish", "fishes"),
    ("walk", "walked", "jump", "jumped"),
    ("play", "played", "watch", "watched"),
    ("open", "opened", "close", "closed"),
    ("run", "ran", "eat", "ate"),
    ("big", "bigger", "small", "smaller"),
    ("good", "better", "bad", "worse"),
    ("day", "days", "night", "nights"),
]


def load(run: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return the raw matrix, the word list, and the row-normalised matrix."""
    embedding = np.load(run / "embeddings.npy")
    words = (run / "vocab.txt").read_text(encoding="utf-8").split("\n")
    if len(words) != embedding.shape[0]:
        raise ValueError(f"{run}: {len(words)} words but {embedding.shape[0]} vectors")
    unit = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding, words, unit


def baseline(unit: np.ndarray, rng: np.random.Generator, draws: int = 5000) -> float:
    """Mean cosine of unrelated pairs: the floor every number has to beat."""
    left = rng.choice(len(unit), draws, replace=False)
    right = rng.choice(len(unit), draws, replace=False)
    return float((unit[left] * unit[right]).sum(axis=1).mean())


def collapse_share(unit: np.ndarray) -> float:
    """Variance share of the first principal component.

    A healthy space spreads its variance; a collapsed one (every vector pointing
    the same way) puts almost all of it on one direction.
    """
    centered = unit - unit.mean(axis=0, keepdims=True)
    spectrum = np.linalg.svd(centered, compute_uv=False)
    share = spectrum**2 / (spectrum**2).sum()
    return float(share[0])


def group_margin(unit: np.ndarray, words: list[str]) -> float:
    """Within-group cosine minus the all-pairs average, over the six groups."""
    index = {word: i for i, word in enumerate(words)}
    picked = [index[w] for group in GROUPS.values() for w in group if w in index]
    block = unit[picked] @ unit[picked].T
    np.fill_diagonal(block, np.nan)
    inside, edge = [], 0
    for members in GROUPS.values():
        size = len(members)
        inside.append(np.nanmean(block[edge:edge + size, edge:edge + size]))
        edge += size
    return float(np.mean(inside) - np.nanmean(block))


def analogies(unit: np.ndarray, words: list[str]) -> dict:
    """Top-1 accuracy and median rank of the expected word, 3CosAdd style."""
    index = {word: i for i, word in enumerate(words)}
    hits, ranks, used, detail = 0, [], 0, []
    for a, b, c, expected in ANALOGIES:
        if not all(word in index for word in (a, b, c, expected)):
            continue
        used += 1
        query = unit[index[b]] - unit[index[a]] + unit[index[c]]
        scores = unit @ query
        scores[[index[a], index[b], index[c]]] = -np.inf
        gave = words[int(np.argmax(scores))]
        hits += gave == expected
        ranks.append(int((scores > scores[index[expected]]).sum()) + 1)
        detail.append({"question": f"{b} - {a} + {c}", "expected": expected, "gave": gave})
    return {
        "questions": used,
        "accuracy": round(hits / used, 3) if used else 0.0,
        "median_rank": float(np.median(ranks)) if ranks else None,
        "analogies_detail": detail,
    }


def neighbours(unit: np.ndarray, words: list[str], k: int = 5) -> dict:
    """Top-k neighbours for every word in the display groups."""
    index = {word: i for i, word in enumerate(words)}
    result = {}
    for word in [w for members in GROUPS.values() for w in members]:
        if word not in index:
            continue
        scores = unit @ unit[index[word]]
        best = np.argsort(-scores)[: k + 1]
        result[word] = [[words[i], round(float(scores[i]), 3)]
                        for i in best if i != index[word]][:k]
    return result


def score(run: Path) -> dict:
    embedding, words, unit = load(run)
    rng = np.random.default_rng(0)
    metrics = {
        "run": run.name,
        "words": len(words),
        "dim": int(embedding.shape[1]),
        "baseline": round(baseline(unit, rng), 3),
        "pc1": round(collapse_share(unit), 3),
        "group_margin": round(group_margin(unit, words), 3),
        **analogies(unit, words),
        "neighbours": neighbours(unit, words),
    }
    config = run / "run.json"
    if config.exists():
        stored = json.loads(config.read_text(encoding="utf-8"))
        for key in ("dim", "window", "negatives", "batch", "lr", "steps"):
            if key in stored:
                metrics[key] = stored[key]
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="directory holding embeddings.npy and vocab.txt")
    args = parser.parse_args()

    metrics = score(args.run)
    print(
        f"{metrics['run']}: baseline {metrics['baseline']:.3f}  pc1 {metrics['pc1']:.3f}  "
        f"group margin {metrics['group_margin']:+.3f}  "
        f"analogies {round(metrics['accuracy'] * metrics['questions'])}/{metrics['questions']}"
    )


if __name__ == "__main__":
    main()
