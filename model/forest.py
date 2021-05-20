from collections import Counter
import numpy as np
from model.tree import CART
import findspark
findspark.init()
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local[8]").setAppName("randomforest")
sc = SparkContext(conf=conf)


def train(tree, X, y):
    rows, cols = X.shape
    sample_ids = np.random.choice(range(rows), size=rows, replace=True)
    if tree.max_features == "sqrt":
        feats_num = int(np.sqrt(cols))
    elif tree.max_features == "log":
        feats_num = int(np.log2(cols))
    elif isinstance(tree.max_features, float):
        feats_num = int(cols * tree.max_features)
    else:
        raise ValueError(f"Can't recognize this max_features: {tree.max_features}")

    feature_ids = np.random.choice(range(cols), size=feats_num, replace=False)
    tree.fit(X[sample_ids][:, feature_ids], y[sample_ids], feature_ids)
    return tree


class RandomForest:
    def __init__(self,
                 n_estimators,
                 min_child_weight=0.1,
                 min_child_nums=2,
                 max_features='sqrt'
                 ):
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.min_child_nums = min_child_nums
        self.max_features = max_features
        self.trees = None
        self.trees_rdd = None

    def fit(self, X, y):
        self.trees = sc.parallelize(
            [
                (CART(self.min_child_weight, self.min_child_nums, self.max_features), X, y)
                for _ in range(self.n_estimators)
            ]
        )
        self.trees = self.trees.map(
            lambda item: train(item[0], item[1], item[2])
        ).collect()

    def predict(self, X):
        assert len(X.shape) == 2
        self.trees_rdd = sc.parallelize([(tree, X) for tree in self.trees])
        pred_rdd = self.trees_rdd.map(lambda item: item[0].predict(item[1])).collect()
        pred_rdd = np.array(pred_rdd).transpose()
        labels = [Counter(x).most_common(1)[0][0] for x in pred_rdd]
        return labels


if __name__ == "__main__":
    from sklearn.metrics import classification_report
    from sklearn.datasets import make_classification
    import time
    rf = RandomForest(100)
    X, y = make_classification(n_samples=200, n_features=50, n_classes=2)
    start = time.time()
    rf.fit(X[:140], y[:140])
    end = time.time()
    print(f"train using {end-start}")
    pred = rf.predict(X[140:])
    end2 = time.time()
    print(f"predict using {end2-end}")
    print(classification_report(y[140:], pred))
