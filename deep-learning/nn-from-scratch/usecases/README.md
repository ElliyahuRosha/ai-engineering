# Use Cases

All use cases for `nn-from-scratch` live directly in this directory. They use
the shared neural-network implementation from the parent directory.

- `xor.py` — executable XOR example
  ([portfolio page](notebooks/xor/)).
- `two_moons.py` — noisy nonlinear binary classification
  ([portfolio page](notebooks/two_moons/)).
- `circles.py` — concentric-circle binary classification
  ([portfolio page](notebooks/circles/)).
- `sine_regression.py` — nonlinear function approximation
  ([portfolio page](notebooks/sine_regression/)).
- `iris.py` — three-class flower classification
  ([portfolio page](notebooks/iris/)).
- `digits_0_vs_1.py` — binary handwritten-digit classification
  ([portfolio page](notebooks/digits_0_vs_1/)).

The scripts expect Python, NumPy, Matplotlib, and scikit-learn to be available.
Dependency installation is intentionally left to the user. Run any use case
directly from this directory, for example:

```bash
cd deep-learning/nn-from-scratch/usecases
python xor.py
python two_moons.py
python circles.py
python sine_regression.py
python iris.py
python digits_0_vs_1.py
```
