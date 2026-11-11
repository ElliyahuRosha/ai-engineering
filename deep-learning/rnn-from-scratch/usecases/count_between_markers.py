import sys
import time
from pathlib import Path

NN_FROM_SCRATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NN_FROM_SCRATCH_DIR))

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from layers.dense import Dense
from layers.rnn import Rnn
from losses import mse, mse_prime
from training import predict


RANDOM_SEED = 7
BASE_DIGITS = 30
SEQUENCE_LENGTH = BASE_DIGITS + 2
TARGET_SCALE = float(BASE_DIGITS)
TRAIN_SAMPLES = 18000
VALIDATION_SAMPLES = 4500
HIDDEN_SIZE = 32
BATCH_SIZE = 300
EPOCHS = 60
LEARNING_RATE = 0.001


def create_dataset(sample_count, seed):
    """Create raw digit sequences and count 0/1 strictly between the 7s."""
    rng = np.random.default_rng(seed)
    sequences = rng.integers(0, 6, size=(sample_count, SEQUENCE_LENGTH))
    targets = np.empty((sample_count, 1, 1), dtype=np.float32)

    for index, sequence in enumerate(sequences):
        left, right = np.sort(
            rng.choice(SEQUENCE_LENGTH, size=2, replace=False)
        )
        sequence[[left, right]] = 7
        between = sequence[left + 1 : right]
        targets[index, 0, 0] = np.count_nonzero(
            (between == 0) | (between == 1)
        )

    # Scale raw digit values to [0, 1], as in the Kaggle notebook.
    X = sequences.astype(np.float32)[:, :, None, None] / 7.0
    return X, targets / TARGET_SCALE, sequences


def adam_update(parameters, gradients, state):
    state["step"] += 1
    for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
        state["first"][index] = 0.9 * state["first"][index] + 0.1 * gradient
        state["second"][index] = 0.999 * state["second"][index] + 0.001 * gradient**2
        first_hat = state["first"][index] / (1 - 0.9**state["step"])
        second_hat = state["second"][index] / (1 - 0.999**state["step"])
        parameter -= LEARNING_RATE * first_hat / (np.sqrt(second_hat) + 1e-8)


def train_batch(network, X_batch, Y_batch, optimizer_state):
    rnn, output_layer = network
    parameters = [rnn.weights_x, rnn.weights_h, rnn.bias, output_layer.weights, output_layer.bias]
    gradients = [np.zeros_like(parameter) for parameter in parameters]
    loss_sum = 0.0
    for sequence, target in zip(X_batch, Y_batch):
        prediction = predict(network, sequence)
        loss_sum += mse(target, prediction)
        hidden_gradient = output_layer.backward(mse_prime(target, prediction), 0.0)
        rnn.backward(hidden_gradient, 0.0)
        sample_gradients = [rnn.dE_dw_x, rnn.dE_dw_h, rnn.dE_db, output_layer.dE_dw, output_layer.dE_db]
        gradients = [total + current for total, current in zip(gradients, sample_gradients)]
    gradients = [gradient / len(X_batch) for gradient in gradients]
    adam_update(parameters, gradients, optimizer_state)
    return loss_sum / len(X_batch)


def make_predictions(network, X, description=None, targets=None):
    """Run the trained network on every sequence in a dataset."""
    starts = range(0, len(X), BATCH_SIZE)
    progress = tqdm(starts, desc=description, unit="batch") if description else None
    predictions, squared_error, seen = [], 0.0, 0
    for start in progress if progress is not None else starts:
        batch = np.array([predict(network, sequence) for sequence in X[start : start + BATCH_SIZE]])
        predictions.append(batch)
        if targets is not None:
            errors = (batch - targets[start : start + BATCH_SIZE]) * TARGET_SCALE
            squared_error += float(np.sum(errors**2))
            seen += len(errors)
            progress.set_postfix(mse=f"{squared_error / seen:.3f}")
    return np.concatenate(predictions)


def regression_metrics(y_true, y_pred):
    """Return MSE, MAE, and R-squared for flattened regression outputs."""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mse_value = float(np.mean((y_true - y_pred) ** 2))
    mae_value = float(np.mean(np.abs(y_true - y_pred)))

    residual_sum = float(np.sum((y_true - y_pred) ** 2))
    total_sum = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2_value = 1.0 - residual_sum / total_sum

    return mse_value, mae_value, r2_value


def fit_with_mse_history(network, X_train, Y_train, X_validation, Y_validation):
    """Train with minibatch Adam and report train/validation MSE each epoch."""
    parameters = [network[0].weights_x, network[0].weights_h, network[0].bias,
                  network[1].weights, network[1].bias]
    optimizer_state = {"step": 0, "first": [np.zeros_like(p) for p in parameters],
                       "second": [np.zeros_like(p) for p in parameters]}
    epoch_history, train_mse_history, validation_mse_history, epoch_durations = [], [], [], []
    training_started = time.perf_counter()

    shuffle_rng = np.random.default_rng(RANDOM_SEED)
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter()
        order = shuffle_rng.permutation(len(X_train))
        progress = tqdm(range(0, len(order), BATCH_SIZE), desc=f"Epoch {epoch:02d}/{EPOCHS} | train", unit="batch")
        for start in progress:
            indices = order[start : start + BATCH_SIZE]
            batch_loss = train_batch(network, X_train[indices], Y_train[indices], optimizer_state)
            seen = min(start + BATCH_SIZE, len(order))
            samples_per_second = seen / max(time.perf_counter() - epoch_started, 1e-8)
            progress.set_postfix(mse=f"{batch_loss * TARGET_SCALE**2:.3f}", samples_s=f"{samples_per_second:.0f}")

        train_predictions = make_predictions(network, X_train)
        validation_predictions = make_predictions(
            network,
            X_validation,
            f"Epoch {epoch:02d}/{EPOCHS} | validation",
            Y_validation,
        )
        train_metrics = regression_metrics(Y_train * TARGET_SCALE, train_predictions * TARGET_SCALE)
        validation_metrics = regression_metrics(Y_validation * TARGET_SCALE, validation_predictions * TARGET_SCALE)
        epoch_history.append(epoch)
        train_mse_history.append(train_metrics[0])
        validation_mse_history.append(validation_metrics[0])
        epoch_durations.append(time.perf_counter() - epoch_started)
        print(f"  train MSE: {train_metrics[0]:.3f} | validation MSE: {validation_metrics[0]:.3f} | "
              f"epoch: {epoch_durations[-1]:.1f}s | average epoch: {np.mean(epoch_durations):.1f}s")

    training_seconds = time.perf_counter() - training_started
    print(f"Training time: {training_seconds:.1f}s")

    history = {
        "epochs": np.array(epoch_history),
        "train_mse": np.array(train_mse_history),
        "validation_mse": np.array(validation_mse_history),
    }

    return (
        training_seconds,
        history,
        train_predictions,
        validation_predictions,
        train_metrics,
        validation_metrics,
    )


def print_configuration():
    print("=" * 72)
    print("COUNT 0s AND 1s BETWEEN TWO MARKERS — RNN FROM SCRATCH")
    print("=" * 72)
    print("Task             : count 0s and 1s strictly between two 7 markers")
    print(f"Sequence         : {BASE_DIGITS} digits from 0..5 + two 7 markers")
    print(f"Total length     : {SEQUENCE_LENGTH}")
    print("Device           : CPU (NumPy implementation)")
    print(f"Training samples : {TRAIN_SAMPLES}")
    print(f"Validation samples: {VALIDATION_SAMPLES}")
    print(f"Architecture     : RNN(1 → {HIDDEN_SIZE}) → Dense({HIDDEN_SIZE} → 1)")
    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Epochs           : {EPOCHS}")
    print(f"Learning rate    : {LEARNING_RATE}")
    print("Optimizer        : Adam")
    print("Output activation: none — this is an unconstrained regression task")
    print("Metrics          : train and validation MSE after every epoch")
    print("=" * 72)


def print_results(training_seconds, train_metrics, validation_metrics):
    train_mse, train_mae, train_r2 = train_metrics
    validation_mse, validation_mae, validation_r2 = validation_metrics

    print("\n" + "=" * 72)
    print("FINAL RESULTS")
    print("=" * 72)
    print(f"Training time: {training_seconds:.2f} seconds")
    print("-" * 72)
    print(f"Train MSE: {train_mse:.8f}")
    print(f"Train MAE: {train_mae:.8f}")
    print(f"Train R²:  {train_r2:.6f}")
    print("-" * 72)
    print(f"Validation MSE: {validation_mse:.8f}")
    print(f"Validation MAE: {validation_mae:.8f}")
    print(f"Validation R²:  {validation_r2:.6f}")
    print("=" * 72)


def print_sample_predictions(
    validation_sequences,
    Y_validation,
    validation_predictions,
    count=8,
):
    print("\nValidation sample predictions")
    print("-" * 72)
    print(" sample | markers | expected | predicted | rounded | abs. error")
    print("-" * 72)

    for index in range(min(count, len(validation_sequences))):
        markers = np.flatnonzero(validation_sequences[index] == 7)
        expected = float(Y_validation[index, 0, 0] * TARGET_SCALE)
        predicted = float(validation_predictions[index, 0, 0] * TARGET_SCALE)
        absolute_error = abs(expected - predicted)

        print(
            f" {index + 1:>6} | {markers[0]:>2},{markers[1]:<2} | "
            f"{expected:>8.0f} | {predicted:>9.3f} | "
            f"{round(predicted):>7} | {absolute_error:>10.3f}"
        )

    print("-" * 72)


def plot_results(history):
    figure, axis = plt.subplots(figsize=(9, 5.5))

    axis.plot(
        history["epochs"],
        history["train_mse"],
        label="Train MSE",
        linewidth=2.2,
    )
    axis.plot(
        history["epochs"],
        history["validation_mse"],
        label="Validation MSE",
        linewidth=2.2,
        linestyle="--",
    )
    axis.set_title("Training and validation MSE")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Count MSE")
    axis.grid(alpha=0.25, which="both")
    axis.legend()

    figure.tight_layout()
    plt.show()


def main():
    np.random.seed(RANDOM_SEED)
    X_train, Y_train, _ = create_dataset(TRAIN_SAMPLES, RANDOM_SEED)
    X_validation, Y_validation, validation_sequences = create_dataset(
        VALIDATION_SAMPLES, RANDOM_SEED + 1
    )

    rnn = Rnn(input_size=1, hidden_size=HIDDEN_SIZE)
    output = Dense(HIDDEN_SIZE, 1)

    # Match torch.nn.Linear's default Kaiming-uniform initialization ranges.
    initializer = np.random.default_rng(RANDOM_SEED)
    input_bound = 1.0 / np.sqrt(1)
    recurrent_bound = 1.0 / np.sqrt(HIDDEN_SIZE)
    output_bound = 1.0 / np.sqrt(HIDDEN_SIZE)
    rnn.weights_x = initializer.uniform(
        -input_bound, input_bound, rnn.weights_x.shape
    ).astype(np.float32)
    rnn.bias = initializer.uniform(
        -input_bound, input_bound, rnn.bias.shape
    ).astype(np.float32)
    rnn.weights_h = initializer.uniform(
        -recurrent_bound, recurrent_bound, rnn.weights_h.shape
    ).astype(np.float32)
    output.weights = initializer.uniform(
        -output_bound, output_bound, output.weights.shape
    ).astype(np.float32)
    output.bias = initializer.uniform(
        -output_bound, output_bound, output.bias.shape
    ).astype(np.float32)

    network = [rnn, output]

    print_configuration()

    (
        training_seconds,
        history,
        train_predictions,
        validation_predictions,
        train_metrics,
        validation_metrics,
    ) = fit_with_mse_history(
        network,
        X_train,
        Y_train,
        X_validation,
        Y_validation,
    )

    print_results(training_seconds, train_metrics, validation_metrics)
    print_sample_predictions(
        validation_sequences,
        Y_validation,
        validation_predictions,
    )
    plot_results(history)


if __name__ == "__main__":
    main()
