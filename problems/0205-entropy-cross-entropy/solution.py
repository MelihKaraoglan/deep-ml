import math

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
    epsilon = 1e-15
    entropy = 0.0
    cross_entropy = 0.0

    for p_i, q_i in zip(P, Q):
        # p_i = 0 ise bu terim 0 sayılır, atla
        if p_i > 0:
            q_i = min(max(q_i, epsilon), 1 - epsilon)
            entropy -= p_i * math.log(p_i)
            cross_entropy -= p_i * math.log(q_i)

    return (entropy, cross_entropy)

