import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
import datetime

def neutralize(alpha: np.array) -> np.array:
    """
        Функция нейтрализации вектора альфы длины N, т.е. сумма 
        элементов вектора равна нулю.

        math:: `\alpha_i = \frac{\Sigma\alpha_i}{N}`,

        Аргументы:
            alpha, np.array - вектор позиций альфы или матрица состояний альф

        Возвращаемое значение:
            alpha_states_neutralized, np.array - нейтрализованный вектор или
                                                 матрица состояний альфы
        
    """
    # if alpha is a vector
    if len(alpha.shape) == 1:
        return alpha - (alpha.sum() / len(alpha))
    else: # if alpha is a matrix of states
        alpha_states_neutralized = np.array([neutralize(_alpha) for _alpha in alpha])
        return alpha_states_neutralized


def normalize(alpha: np.array) -> np.array:
    """
        Функция нормализации вектора альфы длины N, преобразует вектора так,
        что сумма модулей эдементов равна 1.

        math:: `\alpha_i = \frac{\alpha_i}{\Sigma|\alpha_i|}`

        Аргументы:
            alpha, np.array - вектор позиций альфы или матрица состояний альф

        Возвращаемое значение:
            alpha_states_normalized, np.array - нормализованный вектор или
                                                 матрица состояний альфы
        
    """
    # if alpha is a vector
    if len(alpha.shape) == 1:
        return alpha / np.abs(alpha).sum()
    else: # if alpha is a matrix of states
        alpha_states_normalized = np.array([_alpha / np.abs(_alpha).sum() for _alpha in alpha])
        return alpha_states_normalized


def make_alphas_state_matrix_with_day_step(df: pd.DataFrame, days_step: int, strategy=None):
    instruments_number = df.shape[0]
    dates = df.columns[1:]
    alpha_states = np.zeros((df.shape[1]-1, instruments_number))
    for i in range(days_step, len(dates)):
        alpha_states[i] += normalize(neutralize(strategy(df, dates[i - days_step], dates[i-1])))

    return alpha_states


def instrument_return(df=None, close_prices=None):

    """
        Income(\alpha_i) = \frac{\alpha_i(d)}{\alpha_i(d-1)}
    """
    if df is not None:
        # получим матрицу цен закрытия по всем инструментам за каждый день
        close_np = df.drop('Unnamed: 0', axis=1).to_numpy().T

    elif close_prices is not None:
        close_np = close_prices

    # дата начала должна отставать на один день
    close_dates_start, close_dates_end = close_np[:len(close_np)-1], close_np[1:]

    # расчет инкома
    income = close_dates_end / close_dates_start - 1

    return np.concatenate((np.zeros((1, income.shape[1])), income), axis=0)


def alpha_income(alpha_states, return_vector):
    alpha_income_vector = np.zeros(alpha_states.shape[0])
    for i in range(len(alpha_states)-1):
        alpha_income_vector[i] += np.dot(alpha_states[i],return_vector[i+1])
        
    return alpha_income_vector


def turnover(alpha_states_matrix):
    turnover_vec = np.zeros(len(alpha_states_matrix))
    for i in range(1, len(alpha_states_matrix)):
        turnover_vec[i] += np.sum(np.abs(alpha_states_matrix[i] - alpha_states_matrix[i-1]))

    return turnover_vec


def calc_sharpe(alpha_pnl_vec):
    """
        math:: `Sharpe = \sqrt(T)\frac{mean(pnl)}{std(pnl)}`
        math:: `mean(pnl) = \frac{\Sigma_{i=1}^Tpnl_i}{T}`
        math:: `std(pnl) = \sqrt{\frac{1}{T-1}\Sigma_{i=0}^T(pnl_i - mean(pnl))^2}`
    """

    std = np.sqrt(np.sum((alpha_pnl_vec - alpha_pnl_vec.mean())**2) / (len(alpha_pnl_vec) - 1))

    sharpe = len(alpha_pnl_vec)**0.5 * alpha_pnl_vec.mean() / std
    
    return sharpe


def cumulative_pnl(income_vector):
    cumpnl = np.zeros(len(income_vector))
    cumpnl[0] = income_vector[0]

    for i in range(1, len(income_vector)):
        cumpnl[i] = cumpnl[i-1] + income_vector[i]# income_vector[:i+1].sum()

    return cumpnl


def find_drawdown(cumpnl_vec: np.array):
    max_drawdown, max_cumpnl = 0, 0
    drawdown_start, draw_down_end = 0, 0
    
    for _, cumpnl in enumerate(cumpnl_vec):
        if cumpnl >= max_cumpnl:
            max_cumpnl=cumpnl
            # continue
        # else:
        if max_cumpnl - cumpnl >= max_drawdown:
            max_drawdown = max_cumpnl - cumpnl
            drawdown_start, draw_down_end = max_cumpnl, cumpnl

    return max_drawdown, drawdown_start, draw_down_end


def count_instruments_volatility (instruments_incomes):
    
    def std (vector):
        return np.sqrt(np.sum((vector - vector.mean())**2) / (len(vector) - 1))

    # транспонируем матрицу доходностей, потому что сейчас каждая строка - это 
    # доходность каждого инструмента в конкретный день
    volatility = np.array([std(vector) for vector in instruments_incomes.T]) 

    return volatility


def draw_cumpnl(df, cumpnl, dates):
    dates = df.columns[1:]
    plt.plot(cumpnl, label='cumpnl')
    plt.xlabel('date')
    plt.ylabel('cumpnl')
    plt.xticks(np.arange(0, len(dates), 100), dates[np.arange(0, len(dates), 100)], rotation=45)
    plt.legend()
    plt.show()


def AlphaStats(alpha_states, df):
    format = '%Y-%m-%d'
    year_start = '-01-01'
    year_end = '-12-31'

    # get all dates list
    dates = df.columns[1:]

    # get unique years in dates list
    dates_years = np.unique(np.array([date.split('-')[0] for date in dates]))

    # convert date strings into datetime.date to compare
    dates = np.array(pd.to_datetime(dates,format='%Y-%m-%d').date)

    # make years borders
    years_borders = np.array([(datetime.datetime.strptime(year+year_start, format).date(), 
                      datetime.datetime.strptime(year+year_end, format).date()) for year in dates_years])

    # get instruments return matrix 
    return_matrix = instrument_return(df)

    cumpnl = np.array([0])

    annual_cumpnl = []
    annual_sharpe = []
    ave_annual_turnover = []
    annual_drawdown = []

    for idx, year_date in enumerate(years_borders):
        start, end = year_date
        
        # get indexes of the current year
        indexes = np.where((start <= dates) & (end >= dates) , True, False)

        # get alphas for the specific year
        current_year_alpha_states = alpha_states[indexes]

        # get return_vectors for the specific time interval
        current_income_matrix = return_matrix[indexes]

        # get alpha_pnl_vector
        current_alpha_pnl = alpha_income(current_year_alpha_states, current_income_matrix)

        # get cumpnl_vector
        current_alpha_cumpnl_vec = cumulative_pnl(current_alpha_pnl)

        # add the year's cumpnl
        annual_cumpnl.append(current_alpha_cumpnl_vec[-1])

        # add the year's sharpe
        annual_sharpe.append(calc_sharpe(current_alpha_pnl))

        # get turnover for the current_year
        turnover_vec = turnover(current_year_alpha_states)

        # add the current year average turnover
        ave_annual_turnover.append(turnover_vec.sum() / len(turnover_vec))

        # find the current year max drawdown
        drawdown, drawdown_start, drawdown_end = find_drawdown(current_alpha_cumpnl_vec)

        # add the current year drawdown
        annual_drawdown.append(drawdown)

        # concate cumpnl_vec
        cumpnl = np.concatenate((cumpnl, current_alpha_cumpnl_vec + cumpnl[-1]), axis=None)

    # draw cumpnl
    draw_cumpnl(df, cumpnl[1:], df.columns[1:])

    # return annual data as pd.DataFrame
    annual_df = pd.DataFrame({'Year': dates_years,
                             'Sharpe': annual_sharpe,
                             'Average Turnover': ave_annual_turnover,
                             'Max Drawdown': annual_drawdown})

    return annual_df
