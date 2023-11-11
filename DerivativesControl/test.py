import numpy as np

TRESHOLD = 1e-10
YEAR = 'Year'
SHARPE = 'Sharpe'
AVERAGE_TURNOVER = 'Average Turnover'
MAX_DRAWDOWN = 'Max Drawdown'
CUMPNL = 'Cumpnl'

SHARPE_TRESHOLD = 1.0
SHARPE_NUMBER = 3

TURNOVER_TRESHOLD = 0.8


def test1(alpha):
    try:
        if len(alpha.shape) == 1:
            assert abs(alpha.sum()) <= TRESHOLD
        else:
            assert np.where(np.abs(alpha.sum(axis=1)) >= TRESHOLD, 1, 0).sum() == 0
        
        print('Neutrality test passed')
    except AssertionError as e:
        print('Neutrality test is not passed')
    except Exception as e:
        print(f'{e.args}')


def test2(alpha):
    try:
        if len(alpha.shape) == 1:
            if not np.array_equal(alpha, np.zeros_like(alpha)):
                assert abs(np.sum(np.abs(alpha)) - 1.0) <= TRESHOLD
        else:
            assert np.where(abs(np.abs(alpha).sum(axis=1) - 1.0) <= TRESHOLD, 0, 1).sum() == 0
        
        print('Normality test passed')
    except AssertionError as e:
        print('Normality test is not passed')

    except Exception as e:
        print(f'{e.args}')


def test3(alpha_data):
    counter = 0
    try:
        sharpe = alpha_data[SHARPE]

        assert len(np.where(sharpe >= SHARPE_TRESHOLD)[0]) >= SHARPE_NUMBER
        print('Sharpe test passed')
        counter += 1

    except Exception as e:
        print('Sharpe test not passed')

    try:
        turnover = alpha_data[AVERAGE_TURNOVER]

        assert np.average(turnover) < TURNOVER_TRESHOLD
        print('Turnover test passed')
        counter += 1

    except Exception as e:
        print('Turnover tets not passed')

    try:
        cumpnl = alpha_data[CUMPNL]

        assert cumpnl[len(cumpnl)-1] >= 0.3
        print('Cumpnl test passed')
        counter += 1

    except Exception as e:
        print('Cumpnl test not passed')

    print(f'passed {counter} out of {3} tests')

