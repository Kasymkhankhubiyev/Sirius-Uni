import math
import numpy as np


def is_prime(number: int) -> bool:
    if number == 1:
        return False
    elif number == 2:
        return True
    else:
        for i in range(2, round(math.sqrt(number)) + 1):
            if number % i == 0:
                return False
        return True
    

def is_multiple(number: int, devider: int) -> bool:
    figs_sum = 0
    for fig in str(number):
        figs_sum += int(fig)
    if figs_sum % devider == 0:
        return True
    else: return False


def check_first_figure(number: int, starter: int) -> bool:
    if str(number)[0] == str(starter):
        return True
    else: return False


def check_figs_number(number: int, figs_num: int) -> bool:
    if len(str(number)) == figs_num:
        return True
    else: return False


def run(start=1, end=1000, mod=5, first_fig=1, figs_num=2) -> None:
    primes = []
    for number in range(start, end+1, 1):
        if is_prime(number):
            primes.append(number)

    print(primes)

    """
    A = сумма цифр простого числа кратна 5
    В = простое число начинается с 1
    С = простое число двухзначное
    """

    A = np.zeros(len(primes), dtype=int)
    B = np.zeros(len(primes), dtype=int)
    C = np.zeros(len(primes), dtype=int)

    # сохраним соответствующие индексы
    for idx, prime in enumerate(primes):
        if is_multiple(prime, mod):
            A[idx] = 1
        if check_first_figure(prime, first_fig):
            B[idx] = 1
        if check_figs_number(prime, figs_num):
            C[idx] = 1

    print(A.sum(), B.sum(), C.sum())

    A_and_C = np.where((A == 1) & (C == 1), 1, 0)
    A_or_C = np.where((A == 1) | (C == 1), 1, 0)

    A_and_C_and_B = np.where(B == 1, A_and_C, 0)
    A_or_C_and_B = np.where(B == 1, A_or_C, 0)


    A_and_C_except_B = np.where(B == 1, 0, A_and_C)
    A_or_C_except_B = np.where(B == 1, 0, A_or_C)

    print(f'Вероятность события (A & C)B = {A_and_C_and_B.sum()} / {len(primes)} = {A_and_C_and_B.sum() / len(primes)}')
    print(f'Вероятность события (A \/ C)B = {A_or_C_and_B.sum()} / {len(primes)} = {A_or_C_and_B.sum() / len(primes)}')

    print(f'Вероятность события (A & C)\B = {A_and_C_except_B.sum()} / {len(primes)} = {A_and_C_except_B.sum() / len(primes)}')
    print(f'Вероятность события (A \/ C)\B = {A_or_C_except_B.sum()} / {len(primes)} = {A_or_C_except_B.sum() / len(primes)}')




if __name__ == "__main__":
    run()