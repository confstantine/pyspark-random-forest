import numpy as np
from utils import cal_category_gain_gini, cal_numeric_gain_gini
from collections import Counter


MIN_CHILD_WEIGHT = 0.1
MIN_CHILD_NUMS = 3
N_JOBS = 5


class Node:
    def __init__(self, samples, y, feature_ids=None):
        """

        :param samples: numpy.array (n_samples, n_selected_features)
        :param y: numpy.array (n_samples, )
        :param feature_ids: numpy.array maps from split_feature to true idx
        """
        self.samples = samples
        self.y = y
        self.split_feature = None
        self.split = None
        self.children = None
        self.label = None
        self.feature_ids = feature_ids
        assert len(samples.shape) == 2

        is_category = [item.is_integer() for item in samples[0].tolist()]

        self.min_gini = 1
        best_split = None
        for ii in range(samples.shape[1]):
            if is_category[ii]:
                gini = cal_category_gain_gini(samples[:, ii], y)
                if gini < self.min_gini:
                    self.min_gini = gini
                    self.split_feature = ii
            else:
                gini, split = cal_numeric_gain_gini(samples[:, ii], y)
                if gini < self.min_gini:
                    self.min_gini = gini
                    self.split_feature = ii
                    best_split = split

        if not is_category[self.split_feature]:
            self.split = best_split

    def create(self):
        if len(self.samples) < MIN_CHILD_NUMS or self.min_gini < MIN_CHILD_WEIGHT:
            self.label = Counter(self.y).most_common(1)[0][0]
            return
        if self.split:
            self.children = []
            left_idx = self.samples[:, self.split_feature] <= self.split
            self.children.append(Node(
                self.samples[left_idx],
                self.y[left_idx]
            ))
            self.children.append(Node(
                self.samples[~left_idx],
                self.y[~left_idx]
            ))
            for child in self.children:
                child.create()
        else:
            tmp = self.samples[:, self.split_feature]
            categories = np.unique(tmp)
            self.children = {
                cat: Node(
                    self.samples[tmp == cat],
                    self.y[tmp == cat]
                )
                for cat in categories
            }
            for child in self.children.values():
                child.create()
        self.samples = None
        self.y = None

    def predict(self, x):
        if self.feature_ids is not None:
            split_feature = self.feature_ids[self.split_feature]
        else:
            split_feature = self.split_feature
        if self.children:
            if self.split:
                if x[split_feature] <= self.split:
                    return self.children[0].predict(x)
                else:
                    return self.children[1].predict(x)
            else:
                return self.children[x[split_feature]].predict(x)
        else:
            return self.label


class CART:
    def __init__(self):
        self.is_fit = False
        self.node = None
        self.feature_ids = None

    def fit(self, X, y, feature_ids):
        self.node = Node(X, y, feature_ids)
        self.node.create()
        self.is_fit = True
        self.feature_ids = feature_ids

    def predict(self, X):
        pred = []
        for x in X:
            pred.append(self.node.predict(x))
        return pred

    @staticmethod
    def max_depth(self):
        raise NotImplementedError


if __name__ == "__main__":
    import profile
    from sklearn.metrics import classification_report
    import time
    X = np.random.normal(size=(100, 50))
    y = np.random.randint(0, 2, size=(100, ))
    start = time.time()
    cart = CART()
    feature_ids = np.random.choice(range(10), int(np.sqrt(10)), replace=False)
    # profile.run("cart.fit(X[:70, feature_ids], y[:70], feature_ids)")
    cart.fit(X[:70, feature_ids], y[:70], feature_ids)
    end = time.time()
    print(end-start)
    print(classification_report(y[70:], cart.predict(X[70:])))

