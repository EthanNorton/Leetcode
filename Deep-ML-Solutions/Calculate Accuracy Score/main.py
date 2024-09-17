import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy score of a model's predictions.
    
    :param y_true: 1D numpy array containing the true labels.
    :param y_pred: 1D numpy array containing the predicted labels.
    :return: Accuracy score as a float.
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    
    assert len(y_true) == len(y_pred), "Length of true labels and predicted labels must be the same."
    
    
    correct_predictions = np.sum(y_true == y_pred)
    
    
    accuracy = correct_predictions / len(y_true)
    
    return accuracy

