import numpy as np
from collections import Counter
from multiprocessing import Pool


def cal_gini(samples):
    """
    :param
    samples: numpy.array, (n_samples, )
    """
    if len(samples) == 0:
        return 0
    c_k = [v ** 2 for v in Counter(samples).values()]
    return 1 - sum(c_k) / (len(samples) ** 2)


def cal_category_gain_gini(x, y):
    """
    :param x: (n_samples, )
    :param y: (n_samples, )
    :return: tuple: (split, min_gini)
    """

    list_gini = []
    categories = np.unique(x)
    for cat in categories:
        chosens = y[x == cat]
        gini = cal_gini(chosens)
        nums = len(chosens)
        list_gini.append(gini * nums)
    return sum(list_gini) / len(x)


def cal_numeric_gain_gini(x, y):
    min_gini = 1
    best_split = None
    for split in np.linspace(x.min(), x.max(), 50, endpoint=True):
        left_idx = x <= split
        left = y[left_idx]
        right = y[~left_idx]
        gini = (len(left) * cal_gini(left) + len(right) * cal_gini(right)) / len(x)
        if gini < min_gini:
            min_gini = gini
            best_split = split

    return min_gini, best_split


if __name__ == "__main__":
    import time
    # x = np.random.randint(0, 5, size=(100, ))
    x = np.random.normal(size=(100,))
    y = np.random.randint(0, 2, size=(100, ))
    start = time.process_time()
    print(cal_numeric_gain_gini(x, y))
    end = time.process_time()
    print(end-start)
