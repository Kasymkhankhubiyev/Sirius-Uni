import numpy as np
from helper import normalize, neutralize_with_dropout


def truncate_alpha(alpha, treshold=0.1):

    # we take it a little smaller in advance
    _treshold = treshold * 0.9

    iter = 0
    
    while len(np.where(np.abs(alpha) > treshold)[0]) and iter <= 100: # while there is at list one
        iter += 1
        alpha = np.where(np.abs(alpha) > _treshold, _treshold * np.sign(alpha), alpha)
    
        sum_long = alpha[np.where(alpha > 0, True, False)].sum()
        sum_short = - alpha[np.where(alpha < 0, True, False)].sum()
    
        alpha = np.where(alpha > 0, alpha / (2 * sum_long), alpha / (2 * sum_short))

    return alpha


def rank (alpha):
    # получаем индексы по которым сортируется массив
    sorted_indexes = np.argsort(alpha)

    for _idx, index in enumerate(sorted_indexes):
        alpha[index] = _idx / (len(alpha) - 1)

    return alpha   


def CutOutliers(alpha, n, make_tf_vec=True):
    # TODO scale it for matrixes.
    if len(alpha.shape) == 1:
        indexes = np.argsort(alpha)

        false_true_vec = np.ones(len(indexes))

        _indexes = np.concatenate((indexes[:n], indexes[len(alpha) - n : len(alpha)]), axis=None)
        
        for idx in _indexes:
            alpha[idx] = 0
            false_true_vec[idx] = 0

        return alpha, false_true_vec
    
    else:
        new_matrix = np.zeros_like(alpha)
        
        for idx, _alpha in enumerate(alpha):
            if not np.array_equal(_alpha, np.zeros_like(_alpha)):
                new_alpha, false_true_vec = CutOutliers(_alpha, n, True)
                new_matrix[idx] += normalize(neutralize_with_dropout(new_alpha, false_true_vec))
               
        return new_matrix


def CutMiddle(alpha, n):

    if len(alpha.shape) == 1:

        indexes = np.argsort(alpha)
        false_true_vec = np.ones(len(indexes))

        borders = [len(alpha) // 2 - n // 2, len(alpha) // 2 + n // 2 + n%2]

        _indexes = indexes[np.arange(borders[0]-1, borders[1]-1)]

        for idx in _indexes:
            alpha[idx] = 0
            false_true_vec[idx] = 0

        return alpha, false_true_vec
    else:
        new_matrix = np.zeros_like(alpha)
        
        for idx, _alpha in enumerate(alpha):
            if not np.array_equal(_alpha, np.zeros_like(_alpha)):
                new_alpha, false_true_vec = CutMiddle(_alpha, n)
                new_matrix[idx] += normalize(neutralize_with_dropout(new_alpha, false_true_vec))
               
        return new_matrix


def calc_alphas_corr(alpha1, alpha2):
    def std (vector):
        return np.sqrt(np.sum((vector - vector.mean())**2) / (len(vector) - 1))

    corr = np.sum((alpha1 - alpha1.mean()) * (alpha2 - alpha2.mean())) / (std(alpha1) * std(alpha2))

    return corr


def decay (alpha_matrix, n): # decrease turnover
    factors = (np.arange(1, n + 2)) / (n+1)

    _new_alpha_states = np.zeros_like(alpha_matrix)

    for idx in range(n+1, len(alpha_matrix)):
        _alpha = alpha_matrix[idx-n-1:idx] * np.array([factors]).T
        if not np.array_equal(_alpha, np.zeros_like(_alpha)):
            _new_alpha_states[idx] += normalize(_alpha.sum(axis=0))
    
    return _new_alpha_states 

