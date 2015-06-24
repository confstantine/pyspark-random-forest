# -*- coding:utf-8 -*-

# CART树的节点
class Node:
	def __init__(self, records):
		# 该节点包含的所有数据集
		# 是一个字典，key=tuple(特征值)，val=label
		self.records = records
		# 该节点的儿子节点
		self.leftChild = None
		self.rightChild = None
		# 当前节点的最佳分裂属性
		self.bestFeature = None
		# 当前节点的最佳分裂值
		self.bestVal = None
		# 记录当前节点是否是叶子节点
		self.isLeaf = None
		# 如果是叶子节点的话要记录label
		self.label = None

	# 判断当前节点是否是叶子节点
	def is_leaf(self):
		oldLabel = None
		for record in self.records:
			thisLabel = record[-1]
			if oldLabel and oldLabel != thisLabel:
				self.isLeaf = False
				return False
			oldLabel = thisLabel
		self.isLeaf = True
		self.label = oldLabel
		return True

if __name__ == "__main__":
	records = [[1, 1], [2,2], [3, 1], [4, 1]]
	root = Node(records)
	print root.gini