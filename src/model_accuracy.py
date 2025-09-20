# src/model_accuracy.py
"""
Utility script to compute simple accuracy metrics for any local model.
Currently a stub; extend if you add a model.
"""
def accuracy(preds, labels):
    if not preds:
        return 0.0
    correct = sum(1 for p,l in zip(preds, labels) if p==l)
    return correct / len(preds)
