import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NN_FROM_SCRATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NN_FROM_SCRATCH_DIR))

from activations import Tanh
from layers.dense import Dense
from losses import mse, mse_prime
from training import predict, train


np.random.seed(42)

# Generate noisy sine data locally. No external dataset download is needed.
X = np.linspace(-2 * np.pi, 2 * np.pi, 300).reshape(-1, 1)
y = np.sin(X[:, 0]) + np.random.normal(0, 0.05, size=len(X))

valid_rows = np.isfinite(X).all(axis=1) & np.isfinite(y)
X, y = X[valid_rows], y[valid_rows]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_nn = X_train_scaled[:, :, np.newaxis]
X_test_nn = X_test_scaled[:, :, np.newaxis]
y_train_nn = y_train.reshape(-1, 1, 1)

# The final Dense layer is intentionally linear for regression.
network = [
    Dense(1, 16),
    Tanh(),
    Dense(16, 16),
    Tanh(),
    Dense(16, 1),
]

train(
    network,
    mse,
    mse_prime,
    X_train_nn,
    y_train_nn,
    epochs=1_200,
    learning_rate=0.01,
    verbose=True,
)

test_predictions = np.array(
    [float(predict(network, x)[0, 0]) for x in X_test_nn]
)
test_mse = mse(y_test, test_predictions)

print(f"Sine regression test MSE: {test_mse:.6f}")

plot_x = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
plot_x_scaled = scaler.transform(plot_x)
plot_y = np.array(
    [predict(network, x.reshape(1, 1))[0, 0] for x in plot_x_scaled]
)

plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], y_train, s=18, alpha=0.45, label="Training data")
plt.scatter(X_test[:, 0], y_test, s=28, alpha=0.65, label="Test data")
plt.plot(plot_x[:, 0], np.sin(plot_x[:, 0]), "k--", linewidth=2, label="True sin(x)")
plt.plot(plot_x[:, 0], plot_y, color="crimson", linewidth=2.5, label="NN prediction")
plt.title(f"Sine Regression — test MSE: {test_mse:.4f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
