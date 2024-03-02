from math import sqrt


if __name__ == "__main__":

    runable = True
    N = 0
    initial_score = 0
    final_score = 66
    positive_points = 1
    negative_points = -1
    positive_term = 0.65
    negative_term = 1 - positive_term


    def check_equation(number):
        if (number * (positive_points * positive_term + negative_points * negative_term)) == (final_score - initial_score):
            result = True
        else:
            result = False

        return result
    
    while runable:
        N = N + 1
        is_equal = check_equation(N)
        if is_equal:
            runable = False
        else:
            runable = True

    print(N)


