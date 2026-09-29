import numpy as np

def compute_cross_entropy_loss(predicted_probs, true_labels, epsilon=1e-15):
    predicted_probs = np.asarray(predicted_probs, dtype=np.float64)
    true_labels = np.asarray(true_labels, dtype=np.float64)

    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)

    per_sample_loss = -np.sum(true_labels * np.log(predicted_probs), axis=1)

    return float(np.mean(per_sample_loss))
