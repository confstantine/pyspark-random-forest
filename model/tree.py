import numpy as np
from utils import cal_category_gain_gini, cal_numeric_gain_gini
from collections import Counter


class Node:
    def __init__(
            self,
            samples,
            y,
            feature_ids=None,
            is_category=None,
            min_child_weight=0.1,
            min_child_nums=2
    ):
        """

        :param samples: numpy.array (n_samples, n_selected_features)
        :param y: numpy.array (n_samples, )
        :param feature_ids: numpy.array maps from split_feature to true idx
        """
        self.samples = samples
        self.y = y
        self.min_child_weight = min_child_weight
        self.min_child_nums = min_child_nums
        self.is_category = is_category

        self.gini = []
        self.split_feature = None
        self.split = None
        self.children = None
        self.label = None
        self.feature_ids = feature_ids
        assert len(samples.shape) == 2

        self.min_gini = 1
        for ii in range(samples.shape[1]):
            if is_category[ii]:
                gini, split = cal_category_gain_gini(samples[:, ii], y)
            else:
                gini, split = cal_numeric_gain_gini(samples[:, ii], y)
            if gini < self.min_gini:
                self.min_gini = gini
                self.split_feature = ii
                self.gini.append(gini)
                self.split = split

    def create(self):
        if len(self.samples) <= self.min_child_nums \
                or self.min_gini <= self.min_child_weight \
                or len(np.unique(self.y)) == 1 \
                or len(np.unique(self.gini)) == 1:
            self.label = Counter(self.y).most_common(1)[0][0]
            return

        if self.is_category[self.split_feature]:
            left_idx = self.samples[:, self.split_feature] == self.split
        else:
            left_idx = self.samples[:, self.split_feature] <= self.split
        self.children = []
        self.children.append(Node(
            self.samples[left_idx],
            self.y[left_idx],
            self.feature_ids,
            self.is_category,
            self.min_child_weight,
            self.min_child_nums
        ))

        self.children.append(Node(
            self.samples[~left_idx],
            self.y[~left_idx],
            self.feature_ids,
            self.is_category,
            self.min_child_weight,
            self.min_child_nums
        ))

        for child in self.children:
            child.create()

        self.samples = None
        self.y = None

    def predict(self, x):
        if self.feature_ids is not None:
            split_feature = self.feature_ids[self.split_feature]
        else:
            split_feature = self.split_feature
        if self.children:
            if not self.is_category[self.split_feature]:
                if x[split_feature] <= self.split:
                    return self.children[0].predict(x)
                else:
                    return self.children[1].predict(x)
            else:
                if x[split_feature] == self.split:
                    return self.children[0].predict(x)
                else:
                    return self.children[1].predict(x)

        else:
            return self.label


class CART:
    def __init__(
            self,
            min_child_weight=0.1,
            min_child_nums=2,
            max_features='sqrt'):
        self.min_child_weight = min_child_weight
        self.min_child_nums = min_child_nums
        self.max_features = max_features
        self.is_fit = False
        self.node = None
        self.feature_ids = None

    def fit(self, X, y, feature_ids, is_category):
        self.node = Node(X, y, feature_ids, is_category, self.min_child_weight, self.min_child_nums)
        self.node.create()
        self.is_fit = True
        self.feature_ids = feature_ids

    def predict(self, X):
        pred = []
        for x in X:
            pred.append(self.node.predict(x))
        return pred

    @property
    def max_depth(self):
        depth = 0
        stack = [self.node]
        while stack:
            for node in stack:
                stack.pop(0)
                if node.children is not None:
                    if isinstance(node.children, list):
                        stack += node.children
                    else:
                        stack += node.children.values()
            depth += 1
        return depth


if __name__ == "__main__":
    import time
    from sklearn.metrics import classification_report
    import pandas as pd

    trains = pd.read_csv("../data/train.csv", header=None, index_col=None)
    is_category = np.array([trains.dtypes[i] != np.float for i in range(len(trains.columns)-1)])
    train_X = trains.iloc[:, :-1].values
    train_y = trains.iloc[:, -1].values
    tests = pd.read_csv("../data/test.csv", header=None, index_col=None)
    test_X = tests.iloc[:, :-1].values
    test_y = tests.iloc[:, -1].values
    start = time.time()
    cart = CART()
    cart.fit(train_X, train_y, list(range(train_X.shape[1])), is_category)
    end = time.time()
    print(end-start)
    print(classification_report(test_y, cart.predict(test_X)))
    print(cart.max_depth)

