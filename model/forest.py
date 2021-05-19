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
    sample_ids = np.unique(np.random.choice(range(rows), size=rows, replace=True))
    feats_num = int(np.sqrt(cols))
    feature_ids = np.random.choice(range(cols), size=feats_num, replace=False)
    tree.fit(X[sample_ids][:, feature_ids], y[sample_ids], feature_ids)
    return tree


class RandomForest:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.trees = None
        self.trees_rdd = None
        # self.trees = sc.parallelize(self.trees)

    def fit(self, X, y):
        self.trees = sc.parallelize([(CART(), X, y) for _ in range(self.n_estimators)])
        self.trees = self.trees.map(lambda item: train(item[0], item[1], item[2])).collect()

    def predict(self, X):
        assert len(X.shape) == 2
        self.trees_rdd = sc.parallelize([(tree, X) for tree in self.trees])
        pred_rdd = self.trees_rdd.map(lambda item: item[0].predict(item[1])).collect()
        pred_rdd = np.array(pred_rdd).transpose()
        labels = [Counter(x).most_common(1)[0][0] for x in pred_rdd]
        return labels


if __name__ == "__main__":
    from sklearn.metrics import classification_report
    import time
    rf = RandomForest(100)
    X = np.random.normal(size=(200, 50))
    y = np.random.randint(0, 2, size=(200, ))
    start = time.time()
    rf.fit(X, y)
    end = time.time()
    print(f"train using {end-start}")
    pred = rf.predict(X)
    end2 = time.time()
    print(f"predict using {end2-end}")
    print(classification_report(y, pred))
    print(classification_report(y, rf.trees[0].predict(X)))
