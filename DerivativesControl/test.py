import numpy as np

TRESHOLD = 1e-4

def test1(alpha):
    try:
        assert abs(alpha.sum() - 1.0) <= TRESHOLD
        print('Neutrality test passed')
    except Exception as e:
        print('Neutrality test is not passed')


def test2(alpha):
    try:
        assert abs(np.sum(np.abs(alpha)) - 1.0) <= TRESHOLD
        print('Normality test passed')
    except Exception as e:
        print('Normality test is not passed')


def test3(alpha):
    try:
        pass
    except Exception as e:
        pass
