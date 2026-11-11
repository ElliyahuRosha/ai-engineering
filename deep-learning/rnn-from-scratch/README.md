# Recurrent Neural Network From Scratch

A small NumPy implementation of a vanilla recurrent neural network, trained by
backpropagation through time. The marker-counting use case predicts how many
`0` and `1` digits occur strictly between two `7` markers.

## Implementation

- [`layers/rnn.py`](layers/rnn.py) — tanh recurrent layer and backpropagation
  through time.
- [`layers/dense.py`](layers/dense.py) — output layer and gradients.
- [`losses.py`](losses.py) — mean squared error and its derivative.
- [`training.py`](training.py) — forward prediction and the basic training loop.
- [`usecases/count_between_markers.py`](usecases/count_between_markers.py) —
  dataset generation and minibatch Adam training in NumPy.
- [`usecases/notebook1e4eb321ce.ipynb`](usecases/notebook1e4eb321ce.ipynb) — the
  corresponding notebook experiment.

## Run

The Python use case needs NumPy, Matplotlib, and tqdm:

```bash
cd deep-learning/rnn-from-scratch
python usecases/count_between_markers.py
```

The notebook is a separate PyTorch-based experiment and needs PyTorch, NumPy,
Matplotlib, and tqdm in the notebook environment.
