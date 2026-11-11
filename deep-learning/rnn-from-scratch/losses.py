import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def softmax_cross_entropy(logits, target_id):
    shifted = logits - np.max(logits)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / np.sum(exp_scores)
    loss = np.log(np.sum(exp_scores)) - shifted[target_id, 0]
    probabilities[target_id, 0] -= 1
    return loss, probabilities
