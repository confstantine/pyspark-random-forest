import numpy as np
from utils import cal_category_gain_gini, cal_numeric_gain_gini
from collections import Counter


class Node:
    def __init__(self, samples, y, feature_ids=None, min_child_weight=0.1, min_child_nums=3):
        """

        :param samples: numpy.array (n_samples, n_selected_features)
        :param y: numpy.array (n_samples, )
        :param feature_ids: numpy.array maps from split_feature to true idx
        """
        self.samples = samples
        self.y = y
        self.min_child_weight = min_child_weight
        self.min_child_nums = min_child_nums

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
        if len(self.samples) < self.min_child_nums or self.min_gini < self.min_child_weight:
            self.label = Counter(self.y).most_common(1)[0][0]
            return
        if self.split:
            self.children = []
            left_idx = self.samples[:, self.split_feature] <= self.split
            self.children.append(Node(
                self.samples[left_idx],
                self.y[left_idx],
                self.feature_ids,
                self.min_child_weight,
                self.min_child_nums
            ))
            self.children.append(Node(
                self.samples[~left_idx],
                self.y[~left_idx],
                self.feature_ids,
                self.min_child_weight,
                self.min_child_nums
            ))
            for child in self.children:
                child.create()
        else:
            tmp = self.samples[:, self.split_feature]
            categories = np.unique(tmp)
            self.children = {
                cat: Node(
                    self.samples[tmp == cat],
                    self.y[tmp == cat],
                    self.feature_ids,
                    self.min_child_weight,
                    self.min_child_nums
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
    def __init__(self, min_child_weight=0.1, min_child_nums=2, max_features='sqrt'):
        self.min_child_weight = min_child_weight
        self.min_child_nums = min_child_nums
        self.max_features = max_features
        self.is_fit = False
        self.node = None
        self.feature_ids = None

    def fit(self, X, y, feature_ids):
        self.node = Node(X, y, feature_ids, self.min_child_weight, self.min_child_nums)
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
    from sklearn.datasets import make_classification
    start = time.time()
    cart = CART()
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2)
    feature_ids = np.random.choice(range(X.shape[1]), int(np.sqrt(X.shape[1])), replace=False)
    # profile.run("cart.fit(X[:70, feature_ids], y[:70], feature_ids)")
    cart.fit(X[:70, feature_ids], y[:70], feature_ids)
    end = time.time()
    print(end-start)
    print(classification_report(y[70:], cart.predict(X[70:])))
    print(cart.max_depth)

