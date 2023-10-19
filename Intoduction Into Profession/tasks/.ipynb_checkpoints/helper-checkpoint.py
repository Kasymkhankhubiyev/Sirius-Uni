import apimoex
import numpy as np
import pandas as pd
import requests 

def get_data(ticker: str, start_date: str, end_date: str, interval=24) -> pd.DataFrame:
    """
    Функция возвращает данные о котивках за указанный период в формате  `pd.DataFrame`

    Аргументы:
        ticker, str - Тикер ценной бумаги, название комании другими словами
        start_date, str - Дата начала отсчета данных в формате: "ГГГГ-ММ-ДД"
        end_date, str - Дата конца отсчета данных в формате: "ГГГГ-ММ-ДД"
        interval, int - размер свечи, по умолчанию равен дневному размеру.
            Принимает следующие целые значения:
            1 (1 минута), 10 (10 минут), 60 (1 час), 24 (1 день),
            7 (1 неделя), 31 (1 месяц), 4 (1 квартал)

        Возвращаемое значение:
        data_frame, pd.DataFrame - дата фрейм, содержищий информацию о котировках:
            начало (begin, date), цена открытия (open, float), 
            цена закрытия (close, float), наивысшая цена (high, float),
            низшая цена (low, float), объем (value, float)
    """
    with requests.Session() as session: # open an internet session
        # get candles
        data = apimoex.get_market_candles(session, security=ticker, start=start_date, end=end_date)
    
    # returns     
    df = pd.DataFrame(data)

    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    _df = df.copy()
    _df['Date'] = pd.to_datetime(_df['begin'])
    _df = _df.drop('begin', axis=1)
    _df = _df.reset_index(drop=True)
    return _df