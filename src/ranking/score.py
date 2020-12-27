import numpy as np


def max_dcg_score(doc_labels):
    t = np.sort(doc_labels)[::-1]
    return np.sum((np.power(2, t) - 1) / np.log2(np.arange(2, len(t) + 2)))


def dcg_score(scores, labels):
    documents = np.arange(len(labels))
    documents_order_according_to_scores = documents[(-scores).argsort()]
    return np.sum((np.power(2, labels[documents_order_according_to_scores]) - 1) / np.log2(documents + 2))
