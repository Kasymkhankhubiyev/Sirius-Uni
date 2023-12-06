import helper
import manipulate_alpha as mpa
import abc
import numpy as np

from typing import NamedTuple

class DataSet:
    def __init__(self, close_df, open_df, high_df, low_df, volume_df) -> None:
        self.close = close_df.drop(close_df.columns[0], axis=1).to_numpy().T
        self.open = open_df.drop(open_df.columns[0], axis=1).to_numpy().T
        self.high = high_df.drop(high_df.columns[0], axis=1).to_numpy().T
        self.low = low_df.drop(low_df.columns[0], axis=1).to_numpy().T
        self.volume = volume_df.drop(volume_df.columns[0], axis=1).to_numpy().T
        self.close_df = close_df


class Alpha:
    def __init__(self, dataset: DataSet) -> None:
        self.close = dataset.close_df
        self.matrix = np.zeros_like(self.close)

    def get_states_matrix(self) -> np.ndarray:
        return self.matrix
    
    def get_alpha_income(self) -> np.ndarray:
        return helper.alpha_income(self.matrix, helper.instrument_return(self.close))
    

class MeanReversionAlpha(Alpha):
    def __init__(self, dataset: DataSet, day_step=5) -> None:
        super().__init__(dataset)
        self.matrix = helper.normalize(helper.neutralize(self.make_mean_reversion_alpha(day_step)))

    def make_mean_reversion_alpha(self, dataset: DataSet, day_step: int):
        alpha = -np.log(dataset.open[day_step:] / dataset.close[:-day_step])
        return np.concatenate((np.zeros((day_step, alpha.shape[1])), alpha))
    

class MomentumAlpha(Alpha):
    def __init__(self, dataset: DataSet, day_step=1) -> None:
        super().__init__(dataset)
        self.matrix = helper.normalize(helper.neutralize(self.make_momentum_alpha(dataset, day_step)))

    def make_momentum_alpha(self, dataset: DataSet, day_step: int):
        alpha = -np.log(dataset.close[day_step:] / dataset.open[day_step:])
        return np.concatenate((np.zeros((day_step, alpha.shape[1])), alpha))


class Alpha3(Alpha):
    def __init__(self, dataset: DataSet, day_step=10) -> None:
        super().__init__(dataset)
        self.matrix = -self.make_alpha_3(dataset, day_step)
        self.matrix, zeros = mpa.CutMiddle(self.matrix, 20)
        self.matrix, zeros_middles = mpa.CutOutliers(self.matrix, 4)
        zeros = helper.merge_zeros(zeros, zeros_middles)
        self.matrix = mpa.truncate_with_drop_out(self.matrix, zeros, 0.003)
        self.matrix = helper.neutralize_with_dropout(self.matrix, zeros)
        self.matrix = helper.normalize(self.matrix)

    def make_alpha_3(self, dataset: DataSet, day_step: int):
        
        _open = np.zeros_like(dataset.open)
        _volume = np.zeros_like(dataset.volume)

        for idx, (x, y) in enumerate(zip(dataset.open, dataset.volume)):
            _open[idx] = mpa.rank(x)
            _volume[idx] = mpa.rank(y)

        alpha = np.zeros((dataset.open.shape))

        for i in range(day_step, len(dataset.volume)):
            alpha[i] = -1 * mpa.ts_correlation(_open[i-day_step: i], _volume[i-day_step: i])

        return alpha


class Alpha4(Alpha):
    def __init__(self, dataset: DataSet) -> None:
        super().__init__(dataset)
        self.matrix = self.make_alpha_4(dataset)

        self.matrix = mpa.decay(self.matrix, 3)
        self.matrix, zeros = mpa.CutMiddle(self.matrix, 10)
        self.matrix, zeros_middles = mpa.CutOutliers(self.matrix, 2)
        zeros = helper.merge_zeros(zeros, zeros_middles)
        self.matrix = mpa.truncate_with_drop_out(self.matrix, zeros, 0.01)
        self.matrix = helper.neutralize_with_dropout(self.matrix, zeros)
        self.matrix = helper.normalize(self.matrix)

    def make_alpha_4(self, dataset: DataSet):
        return - dataset.low
    

class Alpha7(Alpha):
    def __init__(self, dataset: DataSet) -> None:
        super().__init__(dataset)
        self.matrix = helper.normalize(helper.neutralize(mpa.decay(self.make_alpha_7(dataset), 5) + 0.1))

    def make_alpha_7(self, dataset: DataSet):
        
        adv = mpa.average_dayly_volume(dataset.volume, 20)

        close_delta = np.concatenate((np.zeros((7, dataset.close.shape[1])), 
                                      dataset.close[7:] - dataset.close[:-7]))

        alpha = np.zeros_like(dataset.volume)
        for i in range(60+7, len(alpha)):
            alpha[i] = np.where(adv[i] < dataset.volume[i], 
                                -1. * mpa.ts_rank(np.abs(close_delta[i-60:i])) *\
                                np.sign(close_delta)[i], -1.)      
                        
        return alpha


class Alpha8(Alpha):
    def __init__(self, dataset: DataSet, sum_window=5, delay_window=10) -> None:
        super().__init__(dataset)
        self.matrix = self.make_alpha_8(dataset, sum_window, delay_window)
        self.matrix, zeros = mpa.CutOutliers(self.matrix, 2)
        self.matrix = -mpa.truncate_with_drop_out(self.matrix, zeros, 0.003)
        self.matrix = mpa.neutralize_with_dropout(self.matrix, zeros)
        self.matrix = helper.normalize(self.matrix)

    def make_alpha_8(self, dataset: DataSet, sum_window: int, delay_window: int):
        
        return_matrix = helper.instrument_return(dataset.close_df)

        open_sums = np.zeros(dataset.open.shape)
        returns_sums = np.zeros_like(return_matrix)
        for i in range(sum_window, len(open_sums)):
            open_sums[i] = dataset.open[i-sum_window:i].sum(axis=0)
            returns_sums[i] = return_matrix[i-sum_window:i].sum(axis=0)

        multy = returns_sums * open_sums

        alpha = np.zeros(dataset.open.shape)
        for i in range(delay_window, len(alpha)):
            alpha[i] = -1 * mpa.rank(multy[i] - multy[i-delay_window])

        return alpha
    

class Alpha9(Alpha):
    def __init__(self, dataset: DataSet, min_max_window=7, delta_window=1) -> None:
        super().__init__(dataset)
        self.matrix = self.make_alpha_9(dataset, min_max_window, delta_window)
        self.matrix = mpa.decay(self.matrix, 5)
        self.matrix, zeros = mpa.CutOutliers(self.matrix, 4)
        self.matrix = mpa.truncate_with_drop_out(self.matrix, zeros, 0.01)
        self.matrix = helper.neutralize_with_dropout(self.matrix, zeros)
        self.matrix = helper.normalize(self.matrix)

    def make_alpha_9(self, dataset: DataSet, min_max_window: int, delta_window: int):
    
        _close = np.concatenate((np.zeros((1, dataset.close.shape[1])), 
                                 dataset.close[delta_window:] - dataset.close[:-delta_window]))

        alpha = np.zeros_like(dataset.close)
        for i in range(min_max_window+delta_window, len(alpha)):
            alpha[i] = np.where(mpa.ts_min(_close[i-min_max_window: i]) > 0, 
                                _close[i], 
                                np.where(mpa.ts_max(_close[i-min_max_window:i]) < 0, 
                                         _close[i], -1 * dataset.close[i]))

        return alpha


class Alpha8(Alpha):
    def __init__(self, dataset: DataSet, sum_window=5, delay_window=10) -> None:
        super().__init__(dataset)
        self.matrix = self.make_alpha_8(dataset, sum_window, delay_window)

        self.matrix, zeros = mpa.CutOutliers(self.matrix, 2)
        self.matrix = -mpa.truncate_with_drop_out(self.matrix, zeros, 0.003)
        self.matrix = mpa.neutralize_with_dropout(self.matrix, zeros)
        self.matrix = helper.normalize(self.matrix)

    def make_alpha_8(self, dataset: DataSet, sum_window: int, delay_window: int):
        
        return_matrix = helper.instrument_return(self.close)

        open_sums = np.zeros(dataset.open.shape)
        returns_sums = np.zeros_like(return_matrix)
        for i in range(sum_window, len(open_sums)):
            open_sums[i] = dataset.open[i-sum_window:i].sum(axis=0)
            returns_sums[i] = return_matrix[i-sum_window:i].sum(axis=0)

        multy = returns_sums * open_sums

        alpha = np.zeros(dataset.open.shape)
        for i in range(delay_window, len(alpha)):
            alpha[i] = -1 * mpa.rank(multy[i] - multy[i-delay_window])

        return alpha


class Alpha10(Alpha):
    def __init__(self, dataset: DataSet, min_max_window=4, delta_window=1) -> None:
        super().__init__(dataset)
        self.matrix = self.make_alpha_10(dataset, min_max_window, delta_window)

        self.matrix = mpa.decay(self.matrix, 5)
        self.matrix = mpa.truncate_alpha(self.matrix, 0.1)

        self.matrix = helper.neutralize(self.matrix)
        self.matrix = helper.normalize(self.matrix)


    def make_alpha_10(self, dataset: DataSet, min_max_window: int, delta_window: int):

        _close = np.concatenate((np.zeros((1, dataset.close.shape[1])), 
                                 dataset.close[delta_window:] - dataset.close[:-delta_window]))

        sub_alpha = np.zeros_like(dataset.close)

        for i in range(min_max_window+delta_window, len(sub_alpha)):
            sub_alpha[i] = mpa.rank(np.where(mpa.ts_min(_close[i-min_max_window:i]) > 0, 
                                    _close[i], 
                                    np.where(mpa.ts_max(_close[i-min_max_window: i]) < 0, 
                                             _close[i], 
                                             -1 * _close[i])))

        return sub_alpha
    

class Alpha12(Alpha):
    def __init__(self, dataset: DataSet, delta_window=4) -> None:
        super().__init__(dataset)

        self.matrix = -self.make_alpha_12(dataset, delta_window)
        self.matrix = mpa.truncate_alpha(self.matrix, 0.3)
        self.matrix = mpa.decay(self.matrix, 3)

        self.matrix = helper.neutralize(self.matrix)
        self.matrix = helper.normalize(self.matrix)

    def make_alpha_12(self, dataset: DataSet, delta_window: int):

        alpha = np.zeros_like(dataset.close)
        for i in range(delta_window, len(alpha)):
            alpha[i] = np.sign(dataset.volume[i] - dataset.volume[i-delta_window]) *\
                (-1 * dataset.close[i] - dataset.close[i-delta_window])

        return alpha


class Alpha13(Alpha):
    def __init__(self, dataset: DataSet, cov_window=5) -> None:
        super().__init__(dataset)
        self.matrix = -self.make_alpha_13(dataset, cov_window)

        self.matrix, zeros = mpa.CutOutliers(self.matrix, 2)
        self.matrix = mpa.truncate_with_drop_out(self.matrix, zeros, 0.01)
        self.matrix = helper.neutralize_with_dropout(self.matrix, zeros)
        self.matrix = helper.normalize(self.matrix)

    def make_alpha_13(self, dataset: DataSet, cov_window: int):

        alpha = np.zeros_like(dataset.close)
        close_ranked = np.array([mpa.rank(_close) for _close in dataset.close])
        volume_ranked = np.array([mpa.rank(_volume) for _volume in dataset.volume])

        for i in range(cov_window, len(dataset.close)):
            alpha[i] = -1* mpa.rank(mpa.covarience(close_ranked[i-cov_window:i], 
                                                   volume_ranked[i-cov_window:i]))

        return alpha


class Alpha14(Alpha):
    def __init__(self, dataset: DataSet, delta_window=3, cor_window=15) -> None:
        super().__init__(dataset)
        self.matrix = -self.make_alpha_14(dataset, delta_window, cor_window)

        self.matrix = mpa.decay(self.matrix, 20)
        
        self.matrix = helper.neutralize(self.matrix)
        self.matrix = helper.normalize(self.matrix)

    def make_alpha_14(self, dataset: DataSet, delta_window, cor_window):

        returns = helper.instrument_return(dataset.close_df)

        alpha = np.zeros_like(dataset.close)
        for i in range(cor_window, len(alpha)):
            alpha[i] = -1 * mpa.rank(returns[i] - returns[i-delta_window]) *\
                  helper.calc_alphas_corr(dataset.open[i-cor_window:i], dataset.volume[i-cor_window:i])

        return alpha


class Alpha15(Alpha):
    def __init__(self, dataset: DataSet, cor_window=5, sum_window=3) -> None:
        super().__init__(dataset)
        self.matrix = self.make_alpha_15(dataset, cor_window, sum_window)

        self.matrix = mpa.decay(self.matrix, 2)

        self.matrix = helper.neutralize(self.matrix)
        self.matrix = helper.normalize(self.matrix)

    def make_alpha_15(self, dataset: DataSet, cor_window: int, sum_window: int):

        volume_ranked = np.array([mpa.rank(_volume) for _volume in dataset.volume])
        high_ranked = np.array([mpa.rank(_high) for _high in dataset.high])

        ranked_high_volume = np.zeros_like(dataset.volume)
        for i in range(cor_window, len(ranked_high_volume)):
            ranked_high_volume[i] = mpa.rank(mpa.ts_correlation(high_ranked[i-cor_window:i], 
                                                                volume_ranked[i-cor_window:i]))

        alpha = np.zeros_like(dataset.volume)
        for i in range(sum_window + cor_window, len(alpha)):
            alpha[i] = -1 * np.average(ranked_high_volume[i - sum_window:i], axis=0)
            # alpha[i] = -1 * np.sum(ranked_high_volume[i - sum_window:i], axis=0)

        return alpha
    

class Alpha17(Alpha):
    def __init__(self, dataset: DataSet, ts_rank_close_window=10, delta_window=1, 
                 adv_window=20, ts_rank_volume_window=5) -> None:
        super().__init__(dataset)
        self.matrix = helper.normalize(helper.neutralize(self.make_alpha_17(dataset,
                                                                            ts_rank_close_window,
                                                                            delta_window,
                                                                            adv_window,
                                                                            ts_rank_volume_window)))

    def make_alpha_17(self, dataset: DataSet, ts_rank_close_window: int, 
                      delta_window: int, adv_window: int, ts_rank_volume_window: int):

        close_ranked = np.zeros_like(dataset.close)
        for i in range(ts_rank_close_window+1, len(dataset.close)):
            close_ranked[i] = mpa.rank(mpa.ts_rank(dataset.close[i - ts_rank_close_window - 1: i]))

        delta = np.zeros_like(dataset.close)
        for i in range(delta_window+1, len(delta)):
            delta[i] = dataset.close[i-1] - dataset.close[i-1-delta_window]

        delta_close = np.zeros_like(dataset.close)
        for i in range(delta_window, len(delta_close)):
            delta_close[i] = mpa.rank(delta[i] - delta[i-1])

        volume_adv = mpa.average_dayly_volume(dataset.volume, adv_window)
        volume_ranked = np.zeros_like(dataset.volume)
        for i in range(ts_rank_volume_window+1, len(dataset.volume)):
            volume_ranked[i] = mpa.rank(mpa.ts_rank(volume_adv[i-1-ts_rank_volume_window:i-1]))

        alpha = -1 * close_ranked * delta_close * volume_ranked
        return alpha


class Alpha16(Alpha):
    def __init__(self, dataset: DataSet, cov_window=5) -> None:
        super().__init__(dataset)
        self.matrix = -self.make_alpha_16(dataset, cov_window)

        self.matrix = helper.neutralize(self.matrix)
        self.matrix = helper.normalize(self.matrix)

    def make_alpha_16(self, dataset: DataSet, cov_window):

        volume_ranked = np.array([mpa.rank(_volume) for _volume in dataset.volume])
        high_ranked = np.array([mpa.rank(_high) for _high in dataset.high])

        alpha = np.zeros_like(dataset.volume)
        for i in range(cov_window+1, len(alpha)):
            alpha[i] = -1 * mpa.rank(mpa.covarience(high_ranked[i-1-cov_window:i-1], 
                                                    volume_ranked[i-1-cov_window:i-1]))

        return alpha 
    

class Alpha20(Alpha):
    def __init__(self, dataset: DataSet, delay_window=1) -> None:
        super().__init__(dataset)
        self.matrix = self.make_alpha_20(self, dataset, delay_window=1)
        self.matrix = mpa.decay(self.matrix, 3)

        self.matrix = helper.neutralize(self.matrix)
        self.matrix = helper.normalize(self.matrix)

    def make_alpha_20(self, dataset: DataSet, delay_window: int):
        """
            `math: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))`
        """
        alpha = np.zeros_like(dataset.close)
        for i in range(delay_window, len(dataset.close)):
            alpha[i] = -1 * mpa.rank(dataset.open[i] - dataset.high[i-delay_window]) *\
                        mpa.rank(dataset.open[i] - dataset.close[i-delay_window]) *\
                        mpa.rank(dataset.open[i] - dataset.low[i-delay_window])

        return alpha 
    

class Alpha23(Alpha):
    def __init__(self, dataset: DataSet, sum_window=20, delta_window=2) -> None:
        super().__init__(dataset)
        self.matrix = self.make_alpha_23(dataset, sum_window, delta_window)
        self.matrix = mpa.decay(self.matrix, 10)

        self.matrix = helper.neutralize(self.matrix)
        self.matrix = helper.normalize(self.matrix)

    def make_alpha_23(self, dataset: DataSet, sum_window: int, delta_window: int):
        """
            `math: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)`
        """
        alpha = np.zeros_like(dataset.high)
        for i in range(sum_window, len(dataset.high)):
            alpha[i] = np.where(dataset.high[i-sum_window:i].sum(axis=0) / 20 < dataset.high[i], 
                                -1 * (dataset.high[i] - dataset.high[i-delta_window]), 0)

        return alpha


class Alpha28(Alpha):
    def __init__(self, dataset: DataSet, adv_window=20, corr_window=5, a=1) -> None:
        super().__init__(dataset)
        self.matrix = self.make_alpha_28(dataset, adv_window, corr_window, a)
        self.matrix = mpa.decay(self.matrix, 5)
        self.matrix, zeros = mpa.CutOutliers(self.matrix, 10)
        self.matrix, zeros_mid = mpa.CutMiddle(self.matrix, 10)
        zeros = helper.merge_zeros(zeros, zeros_mid)

        self.matrix = helper.neutralize_with_dropout(self.matrix, zeros)
        self.matrix = helper.normalize(self.matrix)

    def make_alpha_28(self, dataset: DataSet, adv_window: int, corr_window: int, a: int):
        """
            `math: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))`
        """
        adv = mpa.average_dayly_volume(dataset.volume, adv_window)
        print(adv)
        alpha = np.zeros_like(dataset.close)
        for i in range(adv_window + corr_window, len(dataset.close)):
            alpha[i] = mpa.ts_correlation(adv[i - corr_window: i], dataset.low[i-corr_window: i-1]) +\
                                        (dataset.high[i] + dataset.low[i]) / 2 - dataset.close[i]

        
        return mpa.scale(alpha, a)