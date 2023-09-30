import numpy as np
import pandas as pd


def reverse_alpha(start_date: str, end_date: str, df=None, close_prices=None) -> np.array:
    """
        Функуция рассчитывает альфы типа Reverse по временному срезу
    """
    if df is not None:
        return - df[end_date] / df[start_date]  # хранить например как матрицу
    elif close_prices is not None:
        pass