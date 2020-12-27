import kaggle
import numpy as np
import pandas as pd


def order_docs(doc_scores, qids):
    assert len(doc_scores) == len(qids)
    doc_ids = np.arange(1, len(doc_scores) + 1)
    ordered_docs = np.zeros(len(doc_scores), dtype=np.int32)
    
    qid_prev = qids[0]
    i_prev = 0

    for i, qid_i in enumerate(qids):
        if qid_i != qid_prev:
            ordered_docs[i_prev:i] = doc_ids[np.argsort(-doc_scores[i_prev:i]) + i_prev]
            i_prev = i
            qid_prev = qid_i
    ordered_docs[i_prev:] = doc_ids[np.argsort(-doc_scores[i_prev:]) + i_prev]
    
    return ordered_docs


def submit(prediction, submission_path):
	sample = pd.read_csv('data/raw/l2r/sample.made.fall.2019')
	docs = order_docs(prediction, sample['QueryId'].values)
	sample['DocumentId'] = docs
	sample.to_csv(submission_path, index=False)

	kaggle.api.competition_submit(submission_path, '', 'learning-to-rank-made-fall-2019')


