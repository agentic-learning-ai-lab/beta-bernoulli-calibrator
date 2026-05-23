import numpy as np
from sklearn.metrics import roc_auc_score

def get_ece(labels, probs, n_bins=10):
    """
    probs: numpy array of predicted probabilities
    labels: numpy array of true labels (0/1)
    n_bins: number of bins
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:  # last bin includes right edge
            mask = (probs >= bins[i]) & (probs <= bins[i+1])
        else:
            mask = (probs >= bins[i]) & (probs < bins[i+1])
            
        if np.any(mask):
            avg_conf = probs[mask].mean()
            avg_acc  = labels[mask].mean()
            ece += np.abs(avg_conf - avg_acc) * mask.mean()
    return ece

def get_metrics(data, pred_col, print_metrics=True):
    actuals = [1 if i == 'yes' else 0 for i in data['resolution']]
    brier_score = np.mean([(p - y) ** 2 for p, y in zip(data[pred_col], actuals)])
    
    predictions = np.array([0 if p <= 0.5 else 1 for p in data[pred_col]])
    acc = sum(predictions == actuals)/len(data)
    
    auc = roc_auc_score(actuals, data[pred_col])
    
    ece = get_ece(np.array(actuals), np.array(data[pred_col]))

    if print_metrics:
        print(f"BS: {round(brier_score, 4)}")
        print(f"Accuracy: {round(acc, 4)}")
        print(f"AUC: {round(auc, 4)}")
        print(f"ECE: {round(ece, 4)}")
        print()

    return {'brier': brier_score, 'accuracy': acc, 'auc': auc, 'ece': ece}

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count