# Kaggle run

`skipgram_kaggle.ipynb` is self-contained: tokeniser, vocabulary, model, training loop,
evaluation and the grid are all inside it. Nothing is imported from this repository.

## Setup (once)

1. Kaggle → **Datasets → New Dataset** → upload `data/tinystories_20m.txt` (103 MB).
   Cell 4 expects it at `/kaggle/input/datasets/elliyahurosha/tinystories-20m`; if you upload
   under a different owner or slug, change the one `DATASET` line in that cell.
2. Kaggle → **Notebooks → New Notebook** → **File → Import Notebook** → this file.
3. Attach the dataset (right panel → *Add data*).
4. Optional: **Settings → Accelerator → GPU**. Measured on this grid: 7–10x faster than 4 CPU
   cores, because the per-step cost is the optimizer update rather than the sampling.

Internet is not needed once the dataset is attached.

## Run

**Run All.** Order matters only in that the corpus cell must come before training.

| stage | time (GPU) | time (CPU) |
|---|---|---|
| load, tokenise, vocabulary | 25 s | 25 s |
| smoke test, 30 steps | 2 s | 3 s |
| 8 configurations, 1.3B pairs | ~25 min | ~4–5 h |

The CPU estimate is measured, not guessed: a dim-100, `negatives=5` run holds ~140k pairs/s on
4 cores, and `negatives=15` costs roughly three times as much per pair because each step scores
16 columns instead of 6. On a GPU the same grid is coffee; on CPU it is most of a working day.

Results land in `runs/<name>/`: `embeddings.npy`, `vocab.txt`, `metrics.json`. The loop skips a
run whose `metrics.json` already exists, so a dead session can be resumed.

## After the run

Download the winning `runs/<name>/embeddings.npy` and `vocab.txt` and replace `outputs/` in the
project. The walkthrough notebook and the 3D map read those two files and nothing else.
