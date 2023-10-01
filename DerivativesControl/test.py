import numpy as np

TRESHOLD = 1e-4

def test1(alpha):
    try:
        if len(alpha.shape) == 1:
            assert abs(alpha.sum()) <= TRESHOLD
        else:
            assert len(np.where(np.abs(alpha.sum(axis=0)) <= TRESHOLD)[0]) == 0
        print('Neutrality test passed')
    except Exception as e:
        print('Neutrality test is not passed')


def test2(alpha):
    try:
        if len(alpha.shape) == 1:
            assert abs(np.sum(np.abs(alpha)) - 1.0) <= TRESHOLD
        else:
            assert len(np.where(np.abs(np.sum(np.abs(alpha), axis=0) - 1.0) <= TRESHOLD)[0]) == 0
        print('Normality test passed')
    except Exception as e:
        print(e.args)
        print('Normality test is not passed')


def test3(alpha):
    try:
        pass
    except Exception as e:
        pass
