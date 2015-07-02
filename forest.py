# coding:utf-8

from pyspark import SparkConf, SparkContext
from tree import tree
from util import log, get_file_path, get_data, get_random_sample, write_result, get_argument

PARAMS = {
	'forestSize' : 100,
	'featuresAmount' : 0.1,
	'sampleAmount' : 1,
        'slavesAmount' : 5,
}

# 对森林中的某棵树进行训练
def train_tree_of_forest(t):
  records = get_random_sample(shared_trainRecords.value, shared_params.value['sampleAmount'])
  t.train(records, shared_params.value['featuresAmount'])
  return t

def predict_tree_of_forest(record):
        myDict = {}
        # 对森林中的每个树进行预测
        for tree in shared_forest.value:
                label = tree.predict(record)
                if not myDict.has_key(label):
                        myDict[label] = 0
                myDict[label] += 1
        myDict = sorted(myDict.iteritems(), key=lambda x:x[1], reverse=True)
        return myDict[0][0]

if __name__ == '__main__':
        args = get_argument()
        PARAMS['forestSize'] = int(args['ta'])
        PARAMS['featuresAmount'] = float(args['fp'])
        # 申明spark变量
        sconf = SparkConf().setMaster('local[%d]' % PARAMS['slavesAmount']).setAppName('Purchase')
        sc = SparkContext(conf=sconf)
        # 将PARAMS设置为广播变量
        shared_params = sc.broadcast(PARAMS)
	# 获取训练数据与预测数据路径
	(trainDataPath, predictDataPath, resultPath) = (args['td'], args['pd'], args['rd'])
	# 读取训练数据
	log('log;data', 'Start to read training data!')
	trainRecords = get_data(trainDataPath)
        # 将trainRecords设置为广播变量
        shared_trainRecords = sc.broadcast(trainRecords)
	log('log;data', 'Amount of train records: ' + str(len(trainRecords)))
	log('log;data', 'Amount of features: ' + str(len(trainRecords[0]) - 1))
	log('log;data', 'Finish reading training data!')
	# 用一个list来储存所有树
	forest = []
	# 根据全局变量中的forestSize来生成树
	log('log;forest', 'Start to train a forest with params as:;forestSize = ' + str(PARAMS['forestSize']))
	log('log;forest', 'featuresAmount  = ' + str(PARAMS['featuresAmount']) + ';sampleAmount = ' + str(PARAMS['sampleAmount']))
        # 向forest中压入一定数量的tree
	for i in xrange(0, PARAMS['forestSize']):
                forest.append(tree())
        # 根据forest数组生成一个rdd
        rdd = sc.parallelize(forest)
        # 对forest中每个tree运行train_tree的操作
        forest = rdd.map(train_tree_of_forest).collect()
        log('log;forest', 'Finish training forest')
	# 开始预测
	# 读取预测数据
	log('log;data', 'Start to read predict data!')
	predictRecords = get_data(predictDataPath)
	log('log;data', 'Amount of predict records: ' + str(len(predictRecords)))
	log('log;data', 'Finish reading predict data!')
	# 用于记录预测结果的列表
	result = []
	# 遍历预测数据
	log('log;forest', 'Start to predict!')
        # 将测试数据集转换成一个rdd
        rdd = sc.parallelize(predictRecords)
        # 将forest设置成共享变量
        shared_forest = sc.broadcast(forest)
        # 对测试数据中的每一条进行预测
        result = rdd.map(predict_tree_of_forest).collect()
	log('log;forest', 'Finish predicting!')
	# 将预测结果写入结果文件中
	log('log;data', 'Start to write the predict result!')
	write_result(resultPath, result)
	log('log;data', 'Finish writing the predict result!')
        log('log;prog', 'Terminal!')
