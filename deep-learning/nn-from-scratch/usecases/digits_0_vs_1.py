import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

NN_FROM_SCRATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NN_FROM_SCRATCH_DIR))

from activations import Sigmoid, Tanh
from layers.dense import Dense
from losses import mse, mse_prime
from training import predict, train


np.random.seed(42)

# load_digits is bundled with scikit-learn; it requires no manual download.
digits_data = load_digits()
binary_mask = np.isin(digits_data.target, [0, 1])
X = digits_data.data[binary_mask].astype(float)
y = digits_data.target[binary_mask].astype(int)
images = digits_data.images[binary_mask]

valid_rows = np.isfinite(X).all(axis=1) & np.isfinite(y)
X, y, images = X[valid_rows], y[valid_rows], images[valid_rows]

# Pixel values in this dataset range from 0 to 16.
X = X / 16.0

indices = np.arange(len(X))
train_indices, test_indices = train_test_split(
    indices,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]
test_images = images[test_indices]

X_train_nn = X_train[:, :, np.newaxis]
X_test_nn = X_test[:, :, np.newaxis]
y_train_nn = y_train.reshape(-1, 1, 1)

network = [
    Dense(64, 32),
    Tanh(),
    Dense(32, 16),
    Tanh(),
    Dense(16, 1),
    Sigmoid(),
]

train(
    network,
    mse,
    mse_prime,
    X_train_nn,
    y_train_nn,
    epochs=180,
    learning_rate=0.03,
    verbose=True,
)

test_scores = np.array(
    [float(predict(network, x)[0, 0]) for x in X_test_nn]
)
test_predictions = (test_scores >= 0.5).astype(int)
test_accuracy = np.mean(test_predictions == y_test)

print(f"Digits 0 vs 1 test accuracy: {test_accuracy:.2%}")

matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["0", "1"],
).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title(f"Confusion Matrix — accuracy: {test_accuracy:.1%}")

# Show several test images together with the model's prediction confidence.
axes[1].axis("off")
sample_count = min(8, len(test_images))
gallery = np.concatenate(test_images[:sample_count], axis=1)
axes[1].imshow(gallery, cmap="gray")
axes[1].set_title(
    "Predictions: "
    + ", ".join(
        f"{prediction} ({score:.2f})"
        for prediction, score in zip(
            test_predictions[:sample_count],
            test_scores[:sample_count],
        )
    ),
    fontsize=9,
)

plt.tight_layout()
plt.show()
