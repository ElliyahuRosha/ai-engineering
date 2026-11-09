"""Train skip-gram with negative sampling on TinyStories.

The defaults (dim 100, window 2, 5 negatives, three passes of 50M pairs) are the
configuration behind the model in `outputs/`, so `python train.py` alone
reproduces it. `--init` warm-starts from a saved matrix, `--resume` continues an
exact checkpoint, `--fresh` overwrites one.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import word_embeddings as we
from skipgram import SkipGram

PAIRS_PER_EPOCH = 50_000_000


def nearest(embedding: np.ndarray, words: list[str], word: str, k: int = 5) -> list[str]:
    """Top-k neighbours by cosine similarity."""
    unit = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    target = words.index(word)
    scores = unit @ unit[target]
    best = np.argsort(-scores)[: k + 1]
    return [f"{words[i]}={scores[i]:.2f}" for i in best if i != target][:k]


def settings(args: argparse.Namespace, vocab_size: int) -> dict:
    """Everything that changes the model or the maths.

    A resumed run must match these or the checkpoint is meaningless. Batch size,
    learning rate and step count are deliberately absent: they can change.
    """
    return {
        "vocab_size": vocab_size,
        "dim": args.dim,
        "window": args.window,
        "negatives": args.negatives,
    }


def save_checkpoint(
    path: Path,
    model: SkipGram,
    optimizer: torch.optim.Optimizer,
    step: int,
    rng: np.random.Generator,
    config: dict,
) -> None:
    """Everything needed to continue exactly where we stopped."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: SkipGram,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    config: dict,
) -> int:
    """Restore model, optimizer and random state; return the last finished step."""
    state = torch.load(path, weights_only=False)
    if state["config"] != config:
        raise ValueError(
            f"checkpoint was trained with {state['config']}, "
            f"this run uses {config}; refusing to load a mismatched model"
        )
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    rng.bit_generator.state = state["numpy_rng"]
    return int(state["step"])


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cpu":
        # PyTorch defaults to 2 threads here; the machine has 4 cores.
        torch.set_num_threads(os.cpu_count() or 1)

    output = Path(args.out)
    checkpoint = output / "checkpoint.pt"
    if checkpoint.exists() and not (args.resume or args.fresh):
        previous = int(torch.load(checkpoint, weights_only=False)["step"])
        raise SystemExit(
            f"{checkpoint} already holds step {previous:,}. "
            f"Pass --resume to continue it, or --fresh to start over and overwrite it."
        )

    tokens = we.tokenize(we.load_corpus(Path(args.corpus)))
    vocab = we.build_vocab(tokens, args.min_count)
    ids = we.encode(tokens, vocab)
    words = list(vocab)
    del tokens

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    probabilities = we.negative_distribution(ids, len(vocab))

    model = SkipGram(len(vocab), args.dim, sparse=device.type == "cpu").to(device)
    # Adam normalises every coordinate to ~lr per step, which collapses all
    # embeddings onto one direction. Adagrad decays per coordinate instead.
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)

    config = settings(args, len(vocab))
    start_step = 0
    if args.init:
        if args.resume:
            raise SystemExit("--init and --resume both load weights; pick one")
        saved = np.load(args.init)
        if saved.shape != (len(vocab), args.dim):
            raise ValueError(
                f"{args.init} holds {saved.shape}, this run expects {(len(vocab), args.dim)}"
            )
        # A warm start has one matrix, not two: both tables begin from it.
        with torch.no_grad():
            weight = torch.as_tensor(saved, dtype=model.center.weight.dtype)
            model.center.weight.copy_(weight)
            model.context.weight.copy_(weight)
        print(f"started from {args.init}", flush=True)
    if args.resume:
        start_step = load_checkpoint(checkpoint, model, optimizer, rng, config)
        print(f"resumed from step {start_step:,}", flush=True)

    steps = args.steps or int(args.epochs * PAIRS_PER_EPOCH / args.batch)
    print(f"tokens={ids.size:,}  vocab={len(vocab):,}  steps={steps:,}", flush=True)
    print(
        f"dim={args.dim}  window={args.window}  negatives={args.negatives}  "
        f"batch={args.batch}  lr={args.lr:g}  device={device}",
        flush=True,
    )

    loss_fn = nn.BCEWithLogitsLoss()
    targets = torch.cat(
        [torch.ones(args.batch, 1), torch.zeros(args.batch, args.negatives)],
        dim=1,
    ).to(device)
    log_every = max(1, steps // 20)

    started = time.perf_counter()
    running = 0.0
    for step in range(start_step + 1, start_step + steps + 1):
        centers, contexts = we.sample_pairs(ids, args.batch, args.window, rng)
        negatives = we.sample_negatives(args.batch, args.negatives, probabilities, rng)
        batch_contexts = np.column_stack([contexts, negatives])

        logits = model(
            torch.as_tensor(centers, dtype=torch.long, device=device),
            torch.as_tensor(batch_contexts, dtype=torch.long, device=device),
        )
        loss = loss_fn(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += float(loss.detach())
        if step % log_every == 0:
            elapsed = time.perf_counter() - started
            done = step - start_step
            print(
                f"step {step:6,}  loss {running / log_every:.4f}  "
                f"{done / elapsed:6.1f} steps/s  "
                f"{done * args.batch / elapsed / 1000:6.1f}k pairs/s  "
                f"{elapsed:5.1f}s",
                flush=True,
            )
            running = 0.0
        if step % args.save_every == 0:
            save_checkpoint(checkpoint, model, optimizer, step, rng, config)

    elapsed = time.perf_counter() - started
    print(f"done: {steps:,} steps in {elapsed:.1f}s")

    embedding = model.center.weight.detach().cpu().numpy()
    for word in ["king", "queen", "dog", "cat"]:
        print(f"{word:6} -> {', '.join(nearest(embedding, words, word))}")

    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "embeddings.npy", embedding)
    (output / "vocab.txt").write_text("\n".join(words), encoding="utf-8")
    (output / "run.json").write_text(
        json.dumps({**vars(args), "vocab_size": len(vocab), "tokens": int(ids.size),
                    "config": config}, indent=2),
        encoding="utf-8",
    )
    save_checkpoint(checkpoint, model, optimizer, start_step + steps, rng, config)
    print(f"saved {output / 'embeddings.npy'}, {output / 'vocab.txt'}, {checkpoint}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("steps_positional", nargs="?", type=int,
                        help="step count, kept for the old `train.py 3000` form")
    parser.add_argument("--corpus", default="data/tinystories_20m.txt")
    parser.add_argument("--out", default="outputs", help="directory for embeddings and checkpoint")
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=3.0, help="1 epoch = 50M pairs")
    parser.add_argument("--steps", type=int, default=0, help="overrides --epochs when set")
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--init", default=None,
                        help="start from a saved embeddings.npy instead of random weights")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.steps_positional:
        args.steps = args.steps_positional
    return args


if __name__ == "__main__":
    main(parse_args())
