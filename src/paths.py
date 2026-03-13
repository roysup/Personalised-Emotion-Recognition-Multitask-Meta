import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR    = os.path.join(_REPO_ROOT, 'data')
RESULTS_DIR = os.path.join(_REPO_ROOT, 'results')

CSV_PATH = os.path.join(DATA_DIR, 'VREED_data_v2.csv')
PKL_PATH = os.path.join(DATA_DIR, 'unique_id_trials_VREED_v2.pkl')