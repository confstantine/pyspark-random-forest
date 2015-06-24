# RandomForest_Spark

分类器为CART树的随机森林，用Spark实现并行

## 程序架构说明
### data
kaggle上面下载的数据集

### docs
+ 待完善的文档
+ 素材文件夹是放一些类图等文件

### forest.py
随机森林主函数
运行该文件并提供参数

### node.py
决策树中节点数据结构

### test
用于测试的文件，可以忽略

### tree.py
CART树的数据结构

### util.py
一些通用函数

## 使用说明
python forest.py train_data_path predict_data_path result_data_path
python forest.py data/train.csv data/test.csv data/result.csv

## 参数说明
+ 每个节点分割时选择了0.15 × 特征值数量个特征
+ 没有限制树的深度，即树会分裂到最底。平均树高25层左右
