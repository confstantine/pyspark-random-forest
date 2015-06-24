# -*- coding:utf-8 -*-

import sys
import random

from node import Node
from util import *

class tree():
	def __init__(self):
		self.nodes = None

	# 训练一棵树，输入参数为该数使用的records
	def train(self, trainRecords, featurePercent):
                thisId = random.SystemRandom().randint(0, 10000)
		# 申明树的根节点
		root = Node(trainRecords)
		# 申明一个变量来记录树的深度
		layerAmount = 1
		# 用一个list来储存node以表示一棵树
		tree = [root]
		# 每次循环从start开始对后面的节点进行分裂
		start = 0
		# 记录是否终止
		isTerminal = None
		log('log;tree', 'Start to train a tree whose id is ' + str(thisId) + '!')
		while isTerminal is None or not isTerminal:
			isTerminal = True
			# log('log;tree', 'Training the ' + str(layerAmount) + '\'s layer')
			end = len(tree)
			for index in range(start, end):
				cNodeIsLeaf = tree[index].is_leaf()
				isTerminal = isTerminal and cNodeIsLeaf
				# 如果当前节点是叶子节点的话不进行分裂
				if cNodeIsLeaf:
					continue
				# 将当前节点分割成两个子节点
				# 特征是随机抽取的
				bestFeature, bestVal, lChild, rChild = split(tree[index], get_random_features(range(0, len(tree[index].records[0]) - 1), featurePercent))
				# log('debug', 'bestFeature = ' + str(bestFeature) + ' bestVal = ' + str(bestVal))
				# log('debug', 'len(left) = ' + str(len(lChild)) + ' len(right) = ' + str(len(rChild)))
				# 如果左子节点不为空则加入树中且设置成父节点的儿子
				if len(lChild) != 0:
					lChild = Node(lChild)
					tree.append(lChild)
					tree[index].leftChild = lChild
					start += 1
				# 如果右子节点不为空则加入树中且设置成父节点的儿子
				if len(rChild) != 0:
					rChild = Node(rChild)
					tree.append(rChild)
					tree[index].rightChild = rChild
					start += 1
				tree[index].bestFeature = bestFeature
				tree[index].bestVal = bestVal
			# 层数递增
			layerAmount += 1
			start = end
		log('log;tree', 'Finish training a ' + str(layerAmount - 1) + ' layers tree whose id is ' + str(thisId) + '!')
		# 将树中所有节点里面的record都删除以释放内存
		for i, node in enumerate(tree):
			tree[i].records = None
		self.nodes = tree
		return tree

	# 给一个预测记录，根据这课树进行预测
	def predict(self, record):
		cNode = self.nodes[0]
		while not cNode.isLeaf:
			if float(record[cNode.bestFeature]) <= cNode.bestVal:
				cNode = cNode.leftChild
			else:
				cNode = cNode.rightChild
		return cNode.label
