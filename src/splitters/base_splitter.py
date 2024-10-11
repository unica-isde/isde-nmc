import numpy as np


def split_data(x, y, tr_fraction=0.5):
    """Split data to create a random tr-ts partition."""
    n, d = x.shape

    # check if y and x have a consistent no. of samples and labels
    n1 = y.size
    assert (n == n1)

    n_tr = int(np.round(n * tr_fraction))

    idx = np.array(range(0, n))  # 0, 1, 2, ..., n-1
    np.random.shuffle(idx)
    idx_tr = idx[0:n_tr]
    idx_ts = idx[n_tr:n]

    xtr = x[idx_tr, :]
    ytr = y[idx_tr]
    xts = x[idx_ts, :]
    yts = y[idx_ts]
    return xtr, ytr, xts, yts
