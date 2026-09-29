import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    scores = np.array(scores)
    max_score = np.max(scores, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(scores - max_score), axis=-1, keepdims=True))
    return (scores - max_score) - log_sum_exp