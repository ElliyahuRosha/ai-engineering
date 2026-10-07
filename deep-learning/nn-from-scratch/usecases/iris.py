import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NN_FROM_SCRATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NN_FROM_SCRATCH_DIR))

from activations import Tanh
from layers.dense import Dense
from losses import mse, mse_prime
from training import predict, train


np.random.seed(42)

# load_iris is bundled with scikit-learn; it requires no manual download.
iris_data = load_iris()
X = iris_data.data.astype(float)
y = iris_data.target.astype(int)

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

# Encode class numbers as three-element target vectors.
y_train_one_hot = np.eye(3)[y_train]

X_train_nn = X_train_scaled[:, :, np.newaxis]
X_test_nn = X_test_scaled[:, :, np.newaxis]
y_train_nn = y_train_one_hot[:, :, np.newaxis]

# Three output neurons represent the three Iris classes. This version uses
# Tanh + MSE so it remains compatible with the current shared implementation.
network = [
    Dense(4, 8),
    Tanh(),
    Dense(8, 3),
    Tanh(),
]

train(
    network,
    mse,
    mse_prime,
    X_train_nn,
    y_train_nn,
    epochs=900,
    learning_rate=0.03,
    verbose=True,
)

test_scores = np.array(
    [predict(network, x).reshape(-1) for x in X_test_nn]
)
test_predictions = np.argmax(test_scores, axis=1)
test_accuracy = np.mean(test_predictions == y_test)

print(f"Iris test accuracy: {test_accuracy:.2%}")

matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1, 2])
ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=iris_data.target_names,
).plot(cmap="Blues", colorbar=False)
plt.title(f"Iris Confusion Matrix — accuracy: {test_accuracy:.1%}")
plt.tight_layout()
plt.show()
