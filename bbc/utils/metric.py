import numpy as np

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