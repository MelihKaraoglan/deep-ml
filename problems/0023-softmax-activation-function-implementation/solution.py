import math

def softmax(scores: list[float]) -> list[float]:

    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in max_score and scores] 

    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    sum_exp = sum(exp_scores)
    
    return [round(s / sum_exp, 4) for s in exp_scores]
    pass