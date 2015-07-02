# -*- coding:utf-8 -*-
import sys
import copy
import random

from datetime import datetime
from node import Node

# 打印日志信息
# lType：日志类型，error，msg等
# cntnt：日志内容，以英语分号(;)分行
def log(lType, cntnt):
	timestamp = datetime.now().strftime('%H:%M:%S')
	lType = lType.split(';')
	t = ''
	for lt in lType:
		t += ('[' + lt + ']')
	cntnts = cntnt.split(';')
	for item in cntnts:
		print t + '[' + timestamp + ']', '\t\t', item

def get_argument():
  args = '\t'.join(sys.argv[1:])
  args = args.split('-')[1:]
  result = {}
  for arg in args:
    cArg = arg.strip().split('\t')
    if len(cArg) != 2:
      log('error', 'There is something wrong with the parameters you provided!')
      exit(0)
    result[cArg[0]] = cArg[1]
  return result

# 打印帮助信息
def help():
	print '''
usage:
  -td\t:\tpath of the training dataset
  -pd\t:\tpath of the predicting dataset
  -rd\t:\tpath of the result
  -ta\t:\tamount of the tree in the forest
  -fp\t:\tpercentage of the features
	'''

# 获取训练样本路径以及测试样本路径
def get_file_path():
	if len(sys.argv) != 4:
		help()
		exit(0)
	else:
		return sys.argv[1:4]

# 读取样本
def get_data(filePath):
	result = []
	for line in open(filePath, 'r'):
		# 跳过第一行
		if 'id' in line:
			continue
		result.append(line.split(',')[1:])
	return result

def write_result(filePath, dataset):
	f = open(filePath, 'w')
	f.write('id,label\n')
	for i, item in enumerate(dataset):
		f.write(str(i) + ',' + str(item.strip()) + '\n')
	f.close()

# 判断树训练的终止条件
def terminal(PARAMS, layerAmount):
	return layerAmount >= PARAMS['maxLayerAmount']

# 获取某个特征对应列以及类型标签
def get_col_and_label(feature, records):
	result = map(lambda x:[x[feature], x[-1]], records)
	return result

# 根据splitVal将right中小于等于splitVal的records放进left中
def getSplit(feature, rIndex, records, splitVal, left, right, lLen, rLen):
	rIndex = len(records) - rLen
	for i in range(rIndex, len(records)):
		record = records[i]
		if float(record[feature].strip()) > splitVal:
			break
		right[record[-1]] -= 1
		if not left.has_key(record[-1]):
			left[record[-1]] = 0.0
		left[record[-1]] += 1
		lLen += 1
		rLen -= 1
	return (left, right, lLen, rLen)
	# rStart = 0
	# for record in right:
	# 	if float(record[feature].strip()) > splitVal:
	# 		break
	# 	rStart += 1
	# 	left.append(record)
	# right = right[rStart:len(right)]
	# return (left, right)

# left是一个record列表，计算该列表中records的gini系数
def cal_gini(label_accumulator, length):
	# label_accumulator = {}
	# for record in records:
	# 	label = record[-1].strip()
	# 	if label not in label_accumulator:
	# 		label_accumulator[label] = 0.0
	# 	label_accumulator[label] += 1
	accu = 0
	for label in label_accumulator:
		accu += pow(label_accumulator[label] / length, 2)
	return 1 - accu

# 根据最佳的特征与分割值切分数据集并返回两个子集
def splitByFeatureAndVal(records, feature, val):
	left = []
	right = []
	for record in records:
		if float(record[feature]) <= float(val):
			left.append(record)
		else:
			right.append(record)
	return (left, right)

# 对参数提供的节点进行分割
def split(node, fIndex):
	bestSplitFeature = None
	bestSplitVal = None
	smallestGini = None
	# 遍历所有特征值
	for i, feature in enumerate(fIndex):
		# log('debug', str(i) + ' feature: ' + str(bestSplitFeature) + ', ' + str(bestSplitVal) + ', ' + str(smallestGini))
		# 申明两个字典来分别记录两个子集中各个标签的个数
		left = {}
		right = {}
		# 记录左边子集包含的记录条数以及右边子集包含的记录条数
		lLen = 0
		rLen = len(node.records)
		# 初始化右边标签的个数
		for r in node.records:
			if not right.has_key(r[-1]):
				right[r[-1]] = 0.0
			right[r[-1]] += 1.0
		# 根据当前feature排序
		records = sorted(node.records, key=lambda x:x[feature])
		# 遍历所有特征值突变的地方
		oldVal = None
		for rIndex, record in enumerate(records):
			thisVal = record[feature]
			# 发现当前特征值突变的地方
			if oldVal is not None and oldVal != thisVal:
				# 根据突变点两边的值得平均值做分裂
				thisSplitVal = (float(oldVal.strip()) + float(thisVal.strip())) / 2
				# 获得分裂后的两个新分组
				left, right, lLen, rLen = getSplit(feature, rIndex, records, thisSplitVal, left, right, lLen, rLen)
				# 计算此次分裂的gini系数
				total = float(len(records))
				cGini = lLen / total * cal_gini(left, lLen) + rLen / total * cal_gini(right, rLen)
				# 如果此次分裂的gini系数小于最小gini系数，则替换
				if smallestGini is None or cGini < smallestGini:
					smallestGini = cGini
					bestSplitFeature = feature
					bestSplitVal = thisSplitVal
			oldVal = thisVal
	# 结束了两层循环后已经找到对于当前数据集records而言最佳的分割特征和分割值
	# 根据最佳分割特征和分割值将当前数据集分割成两个子集
	bestLeft, bestRight = splitByFeatureAndVal(node.records, bestSplitFeature, bestSplitVal)
	return [bestSplitFeature, bestSplitVal, bestLeft, bestRight]

# 无放回的随机采样，用于随机抽取特征值
def get_random_features(data, per):
	return random.SystemRandom().sample(data, int(len(data) * per))

# 有放回的随机采样，用于样本采集
def get_random_sample(data, per):
	sample = []
	for i in xrange(0, int(len(data) * per)):
		sample.append(data[random.SystemRandom().randint(0, len(data) - 1)])
	return sample



