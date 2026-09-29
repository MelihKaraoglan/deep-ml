import math

def calculate_perplexity(probabilities: list[float]) -> float:
    n = len(probabilities)
    avg_neg_log_prob = -sum(math.log(p) for p in probabilities) / n
    
    return math.exp(avg_neg_log_prob)
