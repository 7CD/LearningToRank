import numpy as np
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

from src.ranking.score import max_dcg_score
from src.ranking.utils import group_by_ids


class LambdaMART:
  def __init__(self, n_trees=5, max_depth=8, learning_rate=0.5, dcg_k=-1):
    self.n_trees = n_trees
    self.max_depth = max_depth
    self.learning_rate = learning_rate
    self.trees = []
    # self.k = dcg_k
 
  def fit(self, X, y, query_ids):
    """
    Fits the model on the training data.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
      Feature representation of each document.

    y : array-like of shape (n_samples,)
      Relevance scores for each document in query.
      Must be numeric. Preferably {0, 1, 2, 3, 4}

    query_ids : array-like of shape (n_samples,)
      Query ids for given documents.
      Single query ids must go successively.
    
    Returns
    -------
    self : LambdaMART
      Fitted model.
    """
    assert X.shape[0] == len(y)
    n_samples = X.shape[0]

    y_by_query = group_by_ids(y, query_ids)
    model_scores_by_query = [np.zeros(len(scores)) for scores in y_by_query]
    max_dcg_by_query = [max_dcg_score(scores) for scores in y_by_query]
    # max_dcg_at_k(scores, self.dcg_k)
    
    for k in tqdm(range(self.n_trees)):
      lambdas, w = np.zeros(n_samples), np.zeros(n_samples)
      doc_idx = 0
      
      for y, model_scores, max_DCG in zip(y_by_query, model_scores_by_query, max_dcg_by_query):
        n_docs = len(y)
        doc_ranks_predicted = np.zeros(n_docs, dtype=np.int64) 
        doc_ranks_predicted[(-model_scores).argsort()] = np.arange(n_docs)

        for y_i, s_i, rank_i in zip(y, model_scores, doc_ranks_predicted):
          indices_j = (y != y_i)
          y_j, s_j, rank_j = y[indices_j], model_scores[indices_j], doc_ranks_predicted[indices_j]

          delta_DCG = np.abs(
              (np.power(2, y_i) - np.power(2, y_j)) * 
              (1. / np.log2(rank_i + 2.) - 1. / np.log2(rank_j + 2.))
          )
          rho_i_j = 1. / (1. + np.exp(np.abs(s_i - s_j)))
          lambda_i_j = -rho_i_j * delta_DCG

          lambda_i = (np.sign(y_i - y_j) * lambda_i_j).sum() / max_DCG
          w_i = (rho_i_j * (1 - rho_i_j) * delta_DCG).sum() / max_DCG

          lambdas[doc_idx], w[doc_idx] = lambda_i, w_i
          doc_idx += 1
      
      tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=10)
      tree.fit(X, lambdas)
 
      model_scores = np.concatenate(model_scores_by_query)
      leaf_by_doc_index = tree.apply(X)

      for leaf in set(leaf_by_doc_index):
        one_leaf_docs_indices = np.where(leaf_by_doc_index == leaf)[0]
        gamma_l_k = lambdas[one_leaf_docs_indices].sum() / w[one_leaf_docs_indices].sum()
        tree.tree_.value[leaf] = -gamma_l_k * self.learning_rate
        model_scores[one_leaf_docs_indices] -= gamma_l_k * self.learning_rate
      
      model_scores_by_query = group_by_ids(model_scores, query_ids)
      
      self.trees.append(tree)
  
  def predict(self, X):
    """
    Predict class or regression value for X.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
      Feature representation of each document.
    
    Returns
    -------
    y : array-like of shape (n_samples,)
      The predicted relevance scores.
    """
    model_scores = np.zeros(X.shape[0])
    for tree in self.trees:
      model_scores += tree.predict(X)
    return model_scores
