# =============================
# ARRAY LOADER  (was make_loader() duplicated in si / stl / tlft / pstl)
# =============================

def make_array_loader(X, y, batch_size, shuffle, seed=42):
    """
    Build a DataLoader directly from numpy arrays.
    Replaces the local make_loader() helpers scattered across scripts.

    Parameters
    ----------
    X          : float32 array (N, window_size, channels)
    y          : float32 array (N,)
    batch_size : int
    shuffle    : bool
    seed       : int — generator seed (only used when shuffle=True)

    Returns
    -------
    DataLoader yielding (X_batch, y_batch) tensors
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    X_t = torch.tensor(X.astype('float32'))
    y_t = torch.tensor(y.astype('float32')).reshape(-1, 1)
    ds  = TensorDataset(X_t, y_t)
    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, generator=g)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
