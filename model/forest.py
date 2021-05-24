from collections import Counter
import numpy as np
from model.tree import CART
import findspark
findspark.init()
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local[8]").setAppName("randomforest")
sc = SparkContext(conf=conf)


def train(tree, X, y, is_category):
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
    tree.fit(X[sample_ids][:, feature_ids], y[sample_ids], feature_ids, is_category[feature_ids])
    return tree


class RandomForest:
    def __init__(self,
                 n_estimators,
                 min_child_weight=0.05,
                 min_child_nums=2,
                 max_features='sqrt'
                 ):
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.min_child_nums = min_child_nums
        self.max_features = max_features
        self.n_class = None
        self.trees = None
        self.trees_rdd = None

    def fit(self, X, y, _is_category):
        self.trees = sc.parallelize(
            [
                CART(self.min_child_weight, self.min_child_nums, self.max_features)
                for _ in range(self.n_estimators)
            ]
        )
        _is_category = sc.broadcast(_is_category)
        X = sc.broadcast(X)
        y = sc.broadcast(y)
        self.trees = self.trees.map(
            lambda item: train(item, X.value, y.value, _is_category.value)
        ).collect()
        self.n_class = len(np.unique(y.value))

    def predict(self, X):
        assert len(X.shape) == 2
        self.trees_rdd = sc.parallelize([tree for tree in self.trees])
        X = sc.broadcast(X)
        pred_rdd = self.trees_rdd.map(lambda item: item.predict(X.value)).collect()
        pred_rdd = np.array(pred_rdd).transpose()
        labels = [Counter(x).most_common(1)[0][0] for x in pred_rdd]
        return labels

    def predict_prob(self, X):
        assert len(X.shape) == 2
        X = sc.broadcast(X)
        self.trees_rdd = sc.parallelize([tree for tree in self.trees])
        pred_rdd = self.trees_rdd.map(lambda item: item.predict(X.value)).collect()
        pred_rdd = np.array(pred_rdd).transpose()

        def cal_prob(x):
            prob = []
            cnt = Counter(x)
            for i in range(self.n_class):
                prob.append(cnt.get(i, 0) / self.n_estimators)
            return prob

        return np.array([cal_prob(x) for x in pred_rdd])


if __name__ == "__main__":
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve
    from sklearn.tree import DecisionTreeClassifier
    import pandas as pd
    import matplotlib.pyplot as plt
    import time

    rf = RandomForest(200, max_features=0.5)
    trains = pd.read_csv("../data/train.csv", header=None, index_col=None)
    is_category = np.array([trains.dtypes[i] != np.float for i in range(len(trains.columns)-1)])
    train_X = trains.iloc[:, :-1].values
    train_y = trains.iloc[:, -1].values
    tests = pd.read_csv("../data/test.csv", header=None, index_col=None)
    test_X = tests.iloc[:, :-1].values
    test_y = tests.iloc[:, -1].values

    start = time.time()
    rf.fit(train_X, train_y, is_category)
    end = time.time()
    print(f"train using {end-start}")
    pred = rf.predict(test_X)
    pred_probs = rf.predict_prob(test_X)[:, 1]
    end2 = time.time()
    print(f"predict using {end2-end}")
    print(classification_report(test_y, pred))

    cart = CART()
    cart.fit(train_X, train_y, list(range(train_X.shape[1])), is_category)
    print(classification_report(test_y, cart.predict(test_X)))

    auc = roc_auc_score(test_y, pred_probs)
    fpr, tpr, _ = roc_curve(test_y, pred_probs)
    plt.figure(dpi=320)
    plt.plot(fpr, tpr, color='darkblue',
             lw=2, label='Random Forest ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')

    dt = DecisionTreeClassifier()
    dt.fit(train_X, train_y)
    probs = dt.predict_proba(test_X)
    auc = roc_auc_score(test_y, probs[:, 1])
    fpr, tpr, _ = roc_curve(test_y, probs[:, 1])
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='CART ROC curve (area = %0.2f)' % auc)

    plt.legend(loc="lower right")
    plt.savefig("./roc.png")

