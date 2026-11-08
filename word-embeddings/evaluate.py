"""Score a trained run so different configurations can be compared.

Reads `embeddings.npy` + `vocab.txt` from a run directory, writes `metrics.json`
next to them, and prints a one-line summary. Point it at several runs with
`--compare` to get a single table.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

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
    embedding = np.load(run / "embeddings.npy")
    words = (run / "vocab.txt").read_text(encoding="utf-8").split("\n")
    unit = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding, words, unit


def baseline(unit: np.ndarray, rng: np.random.Generator, draws: int = 5000) -> float:
    """Mean cosine of unrelated pairs: the floor every number has to beat."""
    left = rng.choice(len(unit), draws, replace=False)
    right = rng.choice(len(unit), draws, replace=False)
    return float((unit[left] * unit[right]).sum(axis=1).mean())


def collapse_share(unit: np.ndarray) -> float:
    """Variance share of the first principal component.

    A collapsed space puts almost everything on one direction; a healthy one
    spreads it. Adam on this problem reached 0.98, Adagrad stays around 0.06.
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
    inside = []
    edge = 0
    for members in GROUPS.values():
        size = len(members)
        inside.append(np.nanmean(block[edge:edge + size, edge:edge + size]))
        edge += size
    return float(np.mean(inside) - np.nanmean(block))


def analogies(unit: np.ndarray, words: list[str]) -> dict:
    """Top-1 accuracy and median rank of the expected word, 3CosAdd style."""
    index = {word: i for i, word in enumerate(words)}
    hits, ranks, used = 0, [], 0
    for a, b, c, expected in ANALOGIES:
        if not all(word in index for word in (a, b, c, expected)):
            continue
        used += 1
        query = unit[index[b]] - unit[index[a]] + unit[index[c]]
        scores = unit @ query
        scores[[index[a], index[b], index[c]]] = -np.inf
        best = words[int(np.argmax(scores))]
        hits += best == expected
        ranks.append(int((scores > scores[index[expected]]).sum()) + 1)
    return {
        "questions": used,
        "accuracy": round(hits / used, 3) if used else None,
        "median_rank": float(np.median(ranks)) if ranks else None,
    }


def neighbours(unit: np.ndarray, words: list[str], k: int = 5) -> dict:
    index = {word: i for i, word in enumerate(words)}
    result = {}
    for group, members in GROUPS.items():
        for word in members:
            if word not in index:
                continue
            scores = unit @ unit[index[word]]
            best = np.argsort(-scores)[: k + 1]
            result[word] = {
                "group": group,
                "neighbours": [[words[i], round(float(scores[i]), 3)]
                               for i in best if i != index[word]][:k],
            }
    return result


def score(run: Path) -> dict:
    embedding, words, unit = load(run)
    rng = np.random.default_rng(0)
    floor = baseline(unit, rng)
    inside = group_margin(unit, words)
    metrics = {
        "run": run.name,
        "words": len(words),
        "dim": int(embedding.shape[1]),
        "baseline": round(floor, 3),
        "pc1": round(collapse_share(unit), 3),
        "group_margin": round(inside, 3),
        **analogies(unit, words),
        "neighbours": neighbours(unit, words),
    }
    config_path = run / "run.json"
    if config_path.exists():
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("dim", "window", "negatives", "subsample", "batch", "lr", "steps"):
            if key in stored:
                metrics[key] = stored[key]
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories")
    parser.add_argument("--compare", action="store_true", help="print one table for all runs")
    args = parser.parse_args()

    metrics = [score(run) for run in args.runs]
    columns = ["run", "dim", "window", "negatives", "subsample", "steps",
               "baseline", "pc1", "group_margin", "accuracy", "median_rank"]
    table = pd.DataFrame(metrics)[[c for c in columns if c in metrics[0]]]
    if args.compare or len(metrics) > 1:
        print(table.to_string(index=False))
    for row in metrics:
        print(f"{row['run']}: baseline {row['baseline']:.3f}  pc1 {row['pc1']:.3f}  "
              f"group margin {row['group_margin']:+.3f}  "
              f"analogies {round(row['accuracy'] * row['questions'])}/{row['questions']}  "
              f"median rank {row['median_rank']}")


if __name__ == "__main__":
    main()
