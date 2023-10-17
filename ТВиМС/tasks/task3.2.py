import numpy as np

def run():
    nums = []
    for num in range(3, 37):
        counter = 0
        for i in range(1, 13):
            for j in range(1, 13):
                for k in range(1, 13):
                    if (i + j + k) == num:
                        counter += 1
        nums.append(counter)

    for num in range(3, 37):
        print(f'число {num} получим {nums[num-3]} способами')

    print(f'Всего способов: {sum(nums)}')

    indexes = np.argsort(np.array(nums))[::-1] - 3
    print(indexes)

    for i in range(5):
        print(indexes[i], np.sort(np.array(nums))[::-1][i], 'вероятность = ', 
              f'{np.sort(np.array(nums))[::-1][i]} / {sum(nums)} = ', 
              np.sort(np.array(nums))[::-1][i] / sum(nums))
        




if __name__ == "__main__":
    run()