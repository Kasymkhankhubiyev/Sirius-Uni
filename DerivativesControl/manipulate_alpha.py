import numpy as np
from helper import normalize, neutralize_with_dropout, std


def truncate_alpha(alpha, treshold=0.1):
    """
    
    """
    if len(alpha.shape) == 1:
        iter = 0
        # we take it a little smaller in advance
        _treshold = treshold * 0.9
        
        while np.where(np.abs(alpha) > _treshold, 1, 0).sum() > 0 and iter <= 100:
            iter += 1

            alpha = np.where(np.abs(alpha) > _treshold, _treshold * np.sign(alpha), alpha)
        
            sum_long = alpha[np.where(alpha > 0, True, False)].sum()
            sum_short = -alpha[np.where(alpha < 0, True, False)].sum()
        
            # alpha = np.where(alpha > 0, alpha / (2 * sum_long), alpha / (2 * sum_short))
            for idx, item in enumerate(alpha):
                if item > 0: # 1e-10:
                    alpha[idx] = alpha[idx] / (2 * sum_long)
                elif item < 0: # -1e-10:
                    alpha[idx] = alpha[idx] / (2 * sum_short)

        return alpha
    
    else:
        # while np.where(np.abs(alpha) > _treshold, 1, 0).sum() > 0 and iter <= 100:
        new_alpha = np.zeros_like(alpha)
        for idx, _alpha in enumerate(alpha):
            # if _alpha.sum() > 1e-12:
            new_alpha[idx] = truncate_alpha(_alpha, treshold)

        alpha = new_alpha

        return new_alpha


def rank (_alpha):
    """
    
    """
    # получаем индексы по которым сортируется массив
    alpha = np.zeros_like(_alpha)
    sorted_indexes = np.argsort(alpha)

    for _idx, index in enumerate(sorted_indexes):
        alpha[index] = _idx / (len(alpha) - 1)

    return alpha


def truncate_with_drop_out(alpha, true_false_vec, treshold=0.1):
    """
    
    """
    new_alpha = np.zeros_like(alpha)
    for idx, (_alpha, zero_vec) in enumerate(zip(alpha, true_false_vec)):
        if _alpha.sum() != 0:
            indexes = [i for i in range(len(zero_vec)) if zero_vec[i] == 1]
            __alpha = truncate_alpha(_alpha[indexes], treshold)
            for i in range(len(indexes)):
                _alpha[indexes[i]] = __alpha[i]
            new_alpha[idx] += _alpha

    return new_alpha


def CutOutliers(alpha, n, make_tf_vec=True):
    """
    
    """
    
    if len(alpha.shape) == 1:
        indexes = np.argsort(alpha)

        false_true_vec = np.ones(len(indexes))

        _indexes = np.concatenate((indexes[:n], indexes[len(alpha) - n : len(alpha)]), axis=None)
        
        for idx in _indexes:
            alpha[idx] = 0.
            false_true_vec[idx] = 0

        return alpha, false_true_vec
    
    else:

        new_matrix = np.zeros_like(alpha)
        zero_matrix = np.zeros_like(alpha)
        
        for idx, _alpha in enumerate(alpha):
            if not np.array_equal(_alpha, np.zeros_like(_alpha)):
                new_alpha, false_true_vec = CutOutliers(_alpha, n, True)
                new_matrix[idx] = new_alpha
                zero_matrix[idx] = false_true_vec
               
        return new_matrix, zero_matrix


def CutMiddle(alpha, n):
    """
    
    """

    if len(alpha.shape) == 1:

        # Получаем индексы альф
        indexes = np.argsort(alpha)
        # coздаем вектор в который укажем пропуски
        false_true_vec = np.ones(len(indexes))

        borders = [len(alpha) // 2 - n // 2, len(alpha) // 2 + n // 2 + n%2]

        _indexes = indexes[np.arange(borders[0]-1, borders[1]-1)]

        for idx in _indexes:
            alpha[idx] = 0.
            false_true_vec[idx] = 0

        return alpha, false_true_vec
    
    else:
        new_matrix = np.zeros_like(alpha)
        zero_matrix = np.zeros_like(alpha)
        
        for idx, _alpha in enumerate(alpha):
            if not np.array_equal(_alpha, np.zeros_like(_alpha)):
                new_alpha, false_true_vec = CutMiddle(_alpha, n)
                new_matrix[idx] = new_alpha
                zero_matrix[idx] = false_true_vec
               
        return new_matrix, zero_matrix


def calc_alphas_corr(alpha1, alpha2):
    """
    
    """
    def std (vector):
        return np.sqrt(np.sum((vector - vector.mean())**2) / (len(vector) - 1))

    corr = np.sum((alpha1 - alpha1.mean()) * (alpha2 - alpha2.mean())) / (std(alpha1) * std(alpha2))

    return corr


def decay (alpha_matrix, n):
    """
        Decreases a turnover of a given alpha accounting n days

        `math:`
    """
    
    factors = (np.arange(1, n + 2)) / (n+1)

    _new_alpha_states = alpha_matrix.copy()

    for idx in range(n+1, len(alpha_matrix)):
        _alpha = alpha_matrix[idx-n-1:idx] * np.array([factors]).T
        if not np.array_equal(_alpha, np.zeros_like(_alpha)):
            _new_alpha_states[idx] = _alpha.sum(axis=0)
    
    return _new_alpha_states 


def crop(alpha, treshold=0.1):
    """
        Crops a given alpha with a given position module width called a treshold.
        So the function replaces all the positions which module is greater than treshold with
        a treshold with a corresponding sign

        Arguments:

        Returns:
    """
    if len(alpha.shape) == 1:
        iter = 0

        alpha = np.where(np.abs(alpha) <= treshold, alpha, treshold * np.sign(alpha))

        return alpha
    
    else:
        new_alpha = np.zeros_like(alpha)
        for idx, _alpha in enumerate(alpha):
            new_alpha[idx] += crop(_alpha, treshold)

        return new_alpha


def ts_correlation(x, y):
    """
    
    """
    corr_vec = np.zeros(x.shape[1])

    for idx, (_x, _y) in enumerate(zip(x, y)):
        corr_vec[idx] = calc_alphas_corr(_x, _y)

    return corr_vec


def open_volume_corr_filter(open, volume):
    """
    `math: correlation(open, volume)`
    """
    return ts_correlation(open, volume)


def average_dayly_volume(volume, day_step):
    """
    
    """
    adv = np.zeros_like(volume)

    for i in range(day_step, len(volume)):
        adv[i] = volume[i-day_step:i].mean(axis=0)

    return adv


def ts_min(matrix):
    """
    
    """
    return np.min(matrix, axis=0)


def ts_max(matrix):
    """
    
    """
    return np.max(matrix, axis=0)


def covarience(vector1, vector2):
    """
    
    """
    return np.sum((vector1 - vector1.mean()) * (vector2 - vector2.mean()), axis=0)


def rolling_std(matrix):
    """
    
    """
    std_vector = np.zeros(matrix.shape[-1])
    for i in range(len(std_vector)):
        std_vector[i] = std(matrix.T[i])

    return std_vector


def ts_rank(time_series):
    result = np.zeros(time_series.shape[1])

    for i in range(len(result)):
        result[i] = rank(time_series.T[i])[-1]

    return result


def scale(matrix, a):
    """
    
    """
    for i in range(len(matrix)):
        if np.abs(matrix[i]).sum() != 0:
            matrix[i] = matrix[i] * 2 / np.abs(matrix[i]).sum()
    return matrix
