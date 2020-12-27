import numpy as np


def group_by_ids(a, group_ids):
    a, ids = np.array(a), np.array(group_ids)
    id_bounds_idxs = [0]
    id_bounds_idxs.extend((ids[1:] != ids[:-1]).nonzero()[0] + 1)
    id_bounds_idxs.append(len(ids))
    a_layed_out = []
    for i in range(len(id_bounds_idxs) - 1):
      a_layed_out.append(a[id_bounds_idxs[i] : id_bounds_idxs[i + 1]])
    return a_layed_out
