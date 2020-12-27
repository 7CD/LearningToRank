import numpy as np

from src.ranking.utils import group_by_ids
from src.ranking.score import dcg_score, max_dcg_score


def get_ndcg_score(scores, labels, query_ids):
    scores_grouped = group_by_ids(scores, query_ids)
    labels_grouped = group_by_ids(labels, query_ids)

    dcg = np.array([dcg_score(scores, labels) for scores, labels in zip(scores_grouped, labels_grouped)])
    max_dcg = np.array([max_dcg_score(labels) for labels in labels_grouped])

    ndcg = dcg / max_dcg

    return np.mean(ndcg)


def evaluate(model, X, y, qid):
    pred = model.predict(X)
    return get_ndcg_score(pred, y, qid)
    
    



