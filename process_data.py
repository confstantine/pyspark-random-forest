import pandas as pd

data = pd.read_csv("./transformed.csv", encoding="utf-8", header=None, index_col=0)
for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())

data = data.sample(frac=1)

train_num = int(0.7 * len(data))
data[:train_num].to_csv("./data/train.csv", header=False, index=False)
data[train_num:].to_csv("./data/test.csv", header=False, index=False)

