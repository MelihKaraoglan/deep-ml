import numpy as np

def calculate_perplexity(probabilities: list[float]) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    return float(np.exp(-np.mean(np.log(probs))))