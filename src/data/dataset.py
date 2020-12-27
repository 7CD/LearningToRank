import os
import tarfile

import gdown
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.datasets import load_svmlight_file


def download_dataset(url, path):
    if not os.path.exists(path):
        gdown.download(url, path, quiet=False)


def process_dataset(raw_dataset_path, processed_dataset_dir):
    train_csr_path = os.path.join(processed_dataset_dir, 'train_csr.npz')
    train_y_path = os.path.join(processed_dataset_dir, 'train_y.npy')
    train_qid_path = os.path.join(processed_dataset_dir, 'train_qid.npy')
    test_csr_path = os.path.join(processed_dataset_dir, 'test_csr.npz')
    test_y_path = os.path.join(processed_dataset_dir, 'test_y.npy')
    test_qid_path = os.path.join(processed_dataset_dir, 'test_qid.npy')
    
    if all(map(os.path.exists, [train_csr_path, train_y_path, train_qid_path, \
                                test_csr_path, test_y_path, test_qid_path])):
        return
    
    tar = tarfile.open(raw_dataset_path, "r:gz")
    raw_dataset_dir_path = os.path.dirname(raw_dataset_path)
    tar.extractall(raw_dataset_dir_path)
    tar.close()
    
    extracted_dir_path = os.path.join(raw_dataset_dir_path, 'l2r')
    
    train_data_path = os.path.join(extracted_dir_path, 'train.txt.gz')
    train_csr, train_y, train_qid = load_svmlight_file(train_data_path, query_id=True)
    
    # There are some invalid samples in training data
    docs_by_query = pd.DataFrame({'doc_index' : np.arange(len(train_y)), 
                                  'labels' : train_y, 
                                  'query' : train_qid}, 
                                  index=train_qid)

    good_indexes = []

    for query in set(train_qid):
      try:
        if len(set(docs_by_query.loc[query].values[:, 1])) > 1:
          good_indexes.extend(docs_by_query.loc[query, 'doc_index'].values)
      except:
        continue
      
    train_csr = train_csr[good_indexes]
    train_qid = train_qid[good_indexes]
    train_y = train_y[good_indexes]
    
    test_data_path = os.path.join(extracted_dir_path, 'test.txt.gz')
    test_csr, test_y, test_qid = load_svmlight_file(test_data_path, query_id=True)
    
    save_npz(train_csr_path, train_csr)
    np.save(train_y_path, train_y)
    np.save(train_qid_path, train_qid)
    
    save_npz(test_csr_path, test_csr)
    np.save(test_y_path, test_y)
    np.save(test_qid_path, test_qid)


def get_dataset(url, raw_dataset_path, processed_dataset_dir):
    download_dataset(url, raw_dataset_path)
    
    process_dataset(raw_dataset_path, processed_dataset_dir)
    
    train_csr_path = os.path.join(processed_dataset_dir, 'train_csr.npz')
    train_y_path = os.path.join(processed_dataset_dir, 'train_y.npy')
    train_qid_path = os.path.join(processed_dataset_dir, 'train_qid.npy')
    
    test_csr_path = os.path.join(processed_dataset_dir, 'test_csr.npz')
    test_y_path = os.path.join(processed_dataset_dir, 'test_y.npy')
    test_qid_path = os.path.join(processed_dataset_dir, 'test_qid.npy')
    
    train_csr = load_npz(train_csr_path)
    train_y = np.load(train_y_path)
    train_qid = np.load(train_qid_path)
    
    test_csr = load_npz(test_csr_path)
    test_y = np.load(test_y_path)
    test_qid = np.load(test_qid_path)
    
    return (train_csr, train_y, train_qid), (test_csr, test_y, test_qid)

