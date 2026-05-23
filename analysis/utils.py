import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import beta as beta_dist


# ---- KL between predicted mixture-of-Beta and human histogram ----
NUM_BINS_KL = 25


def _rebin_histogram(hist, new_bins):
    old_bins = len(hist)
    if old_bins % new_bins != 0:
        raise ValueError("new_bins must divide the original number of bins")
    factor = old_bins // new_bins
    return hist.reshape(new_bins, factor).sum(axis=1)


def _prepare_hist_pdf(data, num_bins=NUM_BINS_KL):
    bin_width = 1.0 / num_bins
    pdfs = []
    for _, row in data.iterrows():
        h = np.array(row["forecast_histogram"], dtype=float)
        h_small = _rebin_histogram(h, num_bins)
        s = h_small.sum()
        pdf = h_small / (s * bin_width) if s > 0 else h_small
        pdfs.append(pdf)
    return pdfs, bin_width


def _mixture_pdf(x, alphas, betas, weights):
    pdf = np.zeros_like(x, dtype=float)
    for a, b, w in zip(alphas, betas, weights):
        pdf += w * beta_dist(a, b).pdf(x)
    return pdf


def get_kl(data, num_bins=NUM_BINS_KL):
    """
    KL(predicted mixture || human histogram), discretized to num_bins bins on [0,1].
    Returns mean KL across rows.
    """
    human_pdfs, bin_width = _prepare_hist_pdf(data, num_bins=num_bins)
    bin_edges   = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    kl_ls = []
    for (_, row), human_pdf in zip(data.iterrows(), human_pdfs):
        alphas = np.array(row["alpha"])
        betas  = np.array(row["beta"])
        w      = np.array(row["weight"])

        pred_pdf = _mixture_pdf(bin_centers, alphas, betas, w)
        pred_pdf = pred_pdf * bin_width
        s = pred_pdf.sum()
        if s > 0:
            pred_pdf = pred_pdf / s
        h = np.array(human_pdf, dtype=float)
        hs = h.sum()
        if hs > 0:
            h = h / hs

        eps = 1e-12
        P = np.clip(pred_pdf, eps, 1.0)
        Q = np.clip(h,        eps, 1.0)
        kl_ls.append(float(np.sum(P * (np.log(P) - np.log(Q)))))

    return float(np.mean(kl_ls))


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
