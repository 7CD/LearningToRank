import argparse
import os
import numpy as np
from scipy.sparse import save_npz, load_npz
import yaml

from src.data.split import train_val_split


def data_split(config_path):
    config = yaml.safe_load(open(config_path))
    
    processed_dataset_dir = config['data_load']['processed_dataset_dir']
    
    train_csr_path = os.path.join(processed_dataset_dir, 'train_csr.npz')
    train_y_path = os.path.join(processed_dataset_dir, 'train_y.npy')
    train_qid_path = os.path.join(processed_dataset_dir, 'train_qid.npy')
    
    train_csr = load_npz(train_csr_path)
    train_y = np.load(train_y_path)
    train_qid = np.load(train_qid_path)
    
    test_size = config['data_split']['test_size']
    random_state = config['data_split']['random_state']

    (X_train, y_train, qid_train), (X_val, y_val, qid_val) = train_val_split(train_csr, train_y, train_qid, 
                                                                             test_size, random_state)
    
    
    save_npz(config['data_split']['train_csr_path'], X_train)
    np.save(config['data_split']['y_train_path'], y_train)
    np.save(config['data_split']['qid_train_path'], qid_train)
    
    save_npz(config['data_split']['X_val_path'], X_val)
    np.save(config['data_split']['y_val_path'], y_val)
    np.save(config['data_split']['qid_val_path'], qid_val)
                                                                             

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_split(config_path=args.config)

