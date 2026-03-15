import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, Sampler

# =============================
# SLIDING WINDOW
# =============================

def create_sliding_windows(data, window_size, stride, task_id=None, trial_col='Trial'):
    """
    Unified sliding-window extractor used by all training scripts.

    Parameters
    ----------
    data        : DataFrame with columns ECG, GSR, AR_Rating, VA_Rating, and trial_col
    window_size : int
    stride      : int
    task_id     : int or None — if provided, included in returned task_ids array
    trial_col   : str — column to group by when extracting windows.
                  Use 'Trial' (default) for per-participant scripts where trial
                  numbers are unique within a participant.
                  Use 'trial_global' for pooled scripts (e.g. PSTL) where the
                  same trial number can appear across different participants.

    Returns
    -------
    X           : float32 array (N, window_size, 2)
    y_ar        : float32 array (N,)
    y_va        : float32 array (N,)
    task_ids    : int64  array (N,)  — all zeros if task_id is None
    trial_ids   : int64  array (N,)
    """
    X, y_ar, y_va, task_ids_out, trial_ids_out = [], [], [], [], []

    for trial_id in sorted(data[trial_col].unique()):
        trial = data[data[trial_col] == trial_id].reset_index(drop=True)
        original_len = len(trial)

        # Pad short trials
        if original_len < window_size:
            pad = window_size - original_len
            pad_df = pd.DataFrame({
                'ECG':       [0.0] * pad,
                'GSR':       [0.0] * pad,
                'AR_Rating': [trial['AR_Rating'].iloc[-1]] * pad,
                'VA_Rating': [trial['VA_Rating'].iloc[-1]] * pad,
                'Trial':     [trial_id] * pad,
            })
            trial = pd.concat([trial, pad_df], ignore_index=True)

        trial_len = len(trial)
        last_start = 0

        # Regular windows
        for i in range(0, trial_len - window_size + 1, stride):
            window = trial[['ECG', 'GSR']].iloc[i: i + window_size].values
            label_idx = min(i + window_size - 1, original_len - 1)
            X.append(window)
            y_ar.append(trial['AR_Rating'].iloc[label_idx])
            y_va.append(trial['VA_Rating'].iloc[label_idx])
            task_ids_out.append(task_id if task_id is not None else 0)
            trial_ids_out.append(trial_id)
            last_start = i

        # Tail window (partial stride at end)
        tail_start = last_start + stride
        if tail_start < trial_len:
            valid = trial[['ECG', 'GSR']].iloc[tail_start:].values
            pad_len = window_size - len(valid)
            padded = np.pad(valid, ((0, pad_len), (0, 0)),
                            mode='constant', constant_values=0)
            X.append(padded)
            y_ar.append(trial['AR_Rating'].iloc[original_len - 1])
            y_va.append(trial['VA_Rating'].iloc[original_len - 1])
            task_ids_out.append(task_id if task_id is not None else 0)
            trial_ids_out.append(trial_id)

    return (
        np.array(X,            dtype=np.float32),
        np.array(y_ar,         dtype=np.float32),
        np.array(y_va,         dtype=np.float32),
        np.array(task_ids_out, dtype=np.int64),
        np.array(trial_ids_out,dtype=np.int64),
    )


def build_support_query(task_df, support_trials, query_trials, ar_or_va='ar',
                        seed=42, window_size=2560, stride=1280):
    """
    Build support and query DataLoaders for meta-learning inner-loop adaptation.

    Parameters
    ----------
    task_df         : DataFrame for one participant (must have Trial, ECG, GSR, AR_Rating, VA_Rating)
    support_trials  : list of trial IDs to use as support set
    query_trials    : list of trial IDs to use as query set (may be empty)
    ar_or_va        : 'ar' or 'va'
    seed            : int — generator seed for support loader shuffle
    window_size     : int
    stride          : int

    Returns
    -------
    sup_loader : DataLoader yielding (X, y) batches from support trials
    q_loader   : DataLoader yielding (X, y) batches from query trials
    """
    sup_df = task_df[task_df['Trial'].isin(sorted(support_trials))]
    Xs, yas, yvs, _, _ = create_sliding_windows(sup_df, window_size, stride)
    y_sup = yas if ar_or_va == 'ar' else yvs

    if query_trials:
        qry_df = task_df[task_df['Trial'].isin(sorted(query_trials))]
        Xq, yar, yvr, _, _ = create_sliding_windows(qry_df, window_size, stride)
        y_q = yar if ar_or_va == 'ar' else yvr
    else:
        Xq   = np.empty((0, window_size, 2), dtype=np.float32)
        y_q  = np.empty((0,), dtype=np.float32)

    X_sup_t = torch.tensor(Xs).float()   # shape: (N, window, channels) — model permutes internally
    y_sup_t = torch.tensor(y_sup).float().unsqueeze(1)
    X_q_t   = torch.tensor(Xq).float()   # shape: (N, window, channels)
    y_q_t   = torch.tensor(y_q).float().unsqueeze(1)

    g = torch.Generator()
    g.manual_seed(seed)
    sup_loader = DataLoader(TensorDataset(X_sup_t, y_sup_t),
                            batch_size=8, shuffle=True,
                            generator=g, num_workers=0)
    q_loader   = DataLoader(TensorDataset(X_q_t, y_q_t),
                            batch_size=32, shuffle=False, num_workers=0)
    return sup_loader, q_loader


class BalancedSampler(Sampler):
    """
    Ensures each batch contains exactly one sample per participant.
    Uses a numpy Generator for deterministic, stateful shuffling across epochs.
    """

    def __init__(self, task_ids, present_task_ids, samples_per_task, seed=None):
        self.present_tasks = sorted(set(present_task_ids))
        self.num_tasks     = len(self.present_tasks)
        self.num_batches   = max(samples_per_task[t] for t in self.present_tasks)
        self.rng           = np.random.default_rng(seed)
        self.task_indices  = {t: np.where(task_ids == t)[0] for t in self.present_tasks}

    def __iter__(self):
        indices = []
        for t in self.present_tasks:
            idx = self.task_indices[t]
            if len(idx) < self.num_batches:
                sampled = self.rng.choice(idx, size=self.num_batches, replace=True)
            else:
                sampled = self.rng.permutation(idx)[:self.num_batches]
            indices.append(sampled)

        indices     = np.array(indices).T.flatten()
        batch_order = self.rng.permutation(self.num_batches)

        shuffled = []
        for b in batch_order:
            shuffled.extend(indices[b * self.num_tasks: (b + 1) * self.num_tasks])
        return iter(shuffled)

    def __len__(self):
        return self.num_batches * self.num_tasks



# =============================
# LOADER BUILDERS
# =============================

def make_single_task_loaders(tasks_dict, window_size, stride,
                             label_type, batch_size, seed,
                             split_type='train', shuffle=True):
    """
    Build one DataLoader per participant (used by STL and PSTL per-participant eval).

    Returns
    -------
    loaders : dict {task_idx: DataLoader}
    """
    loaders = {}
    for task_idx in sorted(tasks_dict.keys()):
        task_data = tasks_dict[task_idx]
        X, y_ar, y_va, _, trial_ids = create_sliding_windows(
            task_data, window_size, stride, task_id=task_idx)

        if len(X) == 0:
            print(f"  Warning: no windows for task {task_idx} ({split_type}), skipping.")
            continue

        #X_t     = torch.tensor(X,          dtype=torch.float32).permute(0, 2, 1)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t     = torch.tensor(y_ar if label_type == 'ar' else y_va,
                               dtype=torch.float32).unsqueeze(1)
        tids_t  = torch.tensor(trial_ids,  dtype=torch.long)
        dataset = TensorDataset(X_t, y_t, tids_t)

        if shuffle:
            g = torch.Generator()
            g.manual_seed(seed)
            loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0, generator=g)
        else:
            loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0)
        loaders[task_idx] = loader
    return loaders


def make_mtl_loader(tasks_dict, window_size, stride,
                    label_type, batch_size, seed):
    """
    Combine all participants into one dataset with a BalancedSampler (used by MTL variants).

    Returns
    -------
    loader           : DataLoader
    total_samples    : int
    sampler          : BalancedSampler (exposed so callers can inspect num_batches)
    """
    all_X, all_y, all_task_ids, all_trial_ids = [], [], [], []
    samples_per_task = {}

    for task_idx in sorted(tasks_dict.keys()):
        X, y_ar, y_va, task_ids, trial_ids = create_sliding_windows(
            tasks_dict[task_idx], window_size, stride, task_id=task_idx)

        if len(X) == 0:
            print(f"  Warning: no windows for task {task_idx}, skipping.")
            continue

        all_X.append(X)
        all_y.append(y_ar if label_type == 'ar' else y_va)
        all_task_ids.append(task_ids)
        all_trial_ids.append(trial_ids)
        samples_per_task[task_idx] = len(X)

    all_X        = np.concatenate(all_X,        axis=0)
    all_y        = np.concatenate(all_y,        axis=0)
    all_task_ids = np.concatenate(all_task_ids, axis=0)
    all_trial_ids= np.concatenate(all_trial_ids,axis=0)

    #X_t      = torch.tensor(all_X,         dtype=torch.float32).permute(0, 2, 1)
    X_t = torch.tensor(all_X, dtype=torch.float32)
    y_t      = torch.tensor(all_y,         dtype=torch.float32).unsqueeze(1)
    tids_t   = torch.tensor(all_task_ids,  dtype=torch.long)
    trids_t  = torch.tensor(all_trial_ids, dtype=torch.long)

    dataset = TensorDataset(X_t, y_t, tids_t, trids_t)
    sampler = BalancedSampler(all_task_ids, list(samples_per_task.keys()), samples_per_task, seed=seed)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         sampler=sampler, num_workers=0)
    return loader, len(dataset), sampler


# =============================
# ARRAY LOADER  (was make_loader() duplicated in si / stl / tlft / pstl)
# =============================

def arrays_to_loader(X, y, batch_size, shuffle, seed=42):
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
