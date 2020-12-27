import argparse
import joblib
import os
import numpy as np
from scipy.sparse import save_npz, load_npz
import yaml

from src.ranking.model import LambdaMART


def train_model(config_path):
    config = yaml.safe_load(open(config_path))
    
    n_trees = config['train']['max_depth']
    max_depth = config['train']['max_depth']
    learning_rate = config['train']['learning_rate']
    
    X = load_npz(config['train']['X_path'])
    y = np.load(config['train']['y_path'])
    qid = np.load(config['train']['qid_path'])

    model = LambdaMART(n_trees, max_depth, learning_rate)
    model.fit(X, y, qid)

    model_name = config['model']['model_name']
    models_folder = config['model']['models_folder']

    joblib.dump(
        model,
        os.path.join(models_folder, model_name)
    )


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)

