import argparse
import json
import os

import joblib
import numpy as np
from scipy.sparse import load_npz
import yaml

from src.evaluate.evaluate import evaluate


def evaluate_model(config_path):
    config = yaml.safe_load(open(config_path))
    
    model = joblib.load(config['evaluate']['model_path'])
    
    X = load_npz(config['evaluate']['X_path'])
    y = np.load(config['evaluate']['y_path'])
    qid = np.load(config['evaluate']['qid_path'])


    ndcg = evaluate(model, X, y, qid)

    metrics_path = config['evaluate']['metrics_path']
    
    json.dump(
        obj={'ndcg': ndcg},
        fp=open(metrics_path, 'w')
    )
    print(f'NDCG score file saved to : {metrics_path}')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)

