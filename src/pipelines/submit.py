import argparse

import joblib
from scipy.sparse import load_npz
import yaml

from src.kaggle.utils import submit as _submit


def submit(config_path):
    config = yaml.safe_load(open(config_path))
    
    model = joblib.load(config['submit']['model_path'])
    
    X = load_npz(config['submit']['X_path'])

    pred = model.predict(X)

    _submit(pred, config['submit']['submission_path'])


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    submit(config_path=args.config)

