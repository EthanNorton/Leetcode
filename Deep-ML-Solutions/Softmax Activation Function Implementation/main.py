import math

def softmax(scores: list[float]) -> list[float]:
    exp_scores = [math.exp(score) for score in scores]  
    total = sum(exp_scores)  
    probabilities = [round(score / total, 4) for score in exp_scores] 
    return probabilities
