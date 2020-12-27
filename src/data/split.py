from sklearn.model_selection import GroupShuffleSplit


def train_val_split(train_csr, train_y, train_qid, test_size, random_state):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(train_csr, train_y, train_qid))
    X_train, y_train, qid_train = train_csr[train_idx], train_y[train_idx], train_qid[train_idx]
    X_val, y_val, qid_val = train_csr[val_idx], train_y[val_idx], train_qid[val_idx]
    return (X_train, y_train, qid_train), (X_val, y_val, qid_val)

