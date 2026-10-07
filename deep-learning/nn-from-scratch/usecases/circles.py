import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NN_FROM_SCRATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NN_FROM_SCRATCH_DIR))

from activations import Sigmoid, Tanh
from layers.dense import Dense
from losses import mse, mse_prime
from training import predict, train


np.random.seed(42)

# Generate concentric circles locally. No external dataset download is needed.
X, y = make_circles(
    n_samples=400,
    noise=0.06,
    factor=0.45,
    random_state=42,
)

valid_rows = np.isfinite(X).all(axis=1) & np.isfinite(y)
X, y = X[valid_rows], y[valid_rows]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_nn = X_train_scaled[:, :, np.newaxis]
X_test_nn = X_test_scaled[:, :, np.newaxis]
y_train_nn = y_train.reshape(-1, 1, 1)

network = [
    Dense(2, 12),
    Tanh(),
    Dense(12, 12),
    Tanh(),
    Dense(12, 1),
    Sigmoid(),
]

train(
    network,
    mse,
    mse_prime,
    X_train_nn,
    y_train_nn,
    epochs=800,
    learning_rate=0.05,
    verbose=True,
)

test_scores = np.array(
    [float(predict(network, x)[0, 0]) for x in X_test_nn]
)
test_predictions = (test_scores >= 0.5).astype(int)
test_accuracy = np.mean(test_predictions == y_test)

print(f"Concentric Circles test accuracy: {test_accuracy:.2%}")

x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 160),
    np.linspace(y_min, y_max, 160),
)
grid = np.column_stack((xx.ravel(), yy.ravel()))
grid_scaled = scaler.transform(grid)
grid_scores = np.array(
    [predict(network, point.reshape(2, 1))[0, 0] for point in grid_scaled]
).reshape(xx.shape)

plt.figure(figsize=(8, 7))
plt.contourf(xx, yy, grid_scores, levels=30, cmap="coolwarm", alpha=0.65)
plt.contour(xx, yy, grid_scores, levels=[0.5], colors="black", linewidths=2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="black")
plt.title(f"Concentric Circles — test accuracy: {test_accuracy:.1%}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.axis("equal")
plt.tight_layout()
plt.show()
