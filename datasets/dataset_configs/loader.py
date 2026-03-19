"""
Generic dataset loader — single entry point for all experiment scripts.

Usage
-----
    from dataset_configs.loader import load_dataset

    df, cfg = load_dataset('dssn_em', preserve_trial_order=True)
    # cfg contains: feature_cols, input_dim, window_size, stride,
    #               splits, participant_ids, results_prefix, ...
"""
import sys, os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_REPO_ROOT, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from config import get_dataset_config


def load_dataset(name: str,
                 preserve_trial_order: bool = False,
                 mode: str = 'standard'):
    """
    Load a dataset by name and return (DataFrame, config_dict).

    Parameters
    ----------
    name : str — 'vreed', 'dssn_eq', or 'dssn_em'
    preserve_trial_order : bool — reorder by canonical trial pkl (PSTL/SI scripts)
    mode : str — 'standard' or 'mtml'

    Returns
    -------
    df  : pd.DataFrame — the loaded, cleaned dataframe
    cfg : dict — dataset config with all parameters:
          feature_cols, input_dim, window_size, stride, pstl_batch, stl_batch,
          mtl_batch, splits, results_prefix, participant_ids,
          uw_logvar_lr_ar, uw_logvar_lr_va, ...
    """
    cfg = get_dataset_config(name)

    key = name.lower().replace('-', '_')
    if key == 'vreed':
        from dataset_configs.vreed import load_vreed_df
        df = load_vreed_df(preserve_trial_order=preserve_trial_order, mode=mode)
    elif key == 'dssn_eq':
        from dataset_configs.dssn_eq import load_dssn_eq_df
        df = load_dssn_eq_df(preserve_trial_order=preserve_trial_order, mode=mode)
    elif key == 'dssn_em':
        from dataset_configs.dssn_em import load_dssn_em_df
        df = load_dssn_em_df(preserve_trial_order=preserve_trial_order, mode=mode)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Add derived fields to config
    cfg['participant_ids']    = sorted(cfg['splits'].keys())
    cfg['num_tasks']          = len(cfg['participant_ids'])
    cfg['train_participants'] = sorted(
        [p for p in cfg['participant_ids'] if p not in cfg['test_participants']])

    return df, cfg
