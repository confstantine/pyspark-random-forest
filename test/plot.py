# coding:utf-8
import sys
import matplotlib.pyplot as plt
from numpy.random import rand

color = [('yellow', 0.8), ('blue', 0.3)]

# 将两组点绘制在同一张图上
def plot(aPoints, bPoints):
  points = [aPoints, bPoints]
  for index, point in enumerate(points):
    showOnePlot(index, point)
  plt.legend()
  plt.grid(True)
  plt.show()

# 将一组点绘制在图上
def showOnePlot(index, points):
  plt.scatter(points[0], points[1], c=color[index][0], alpha=color[index][1], s=100, edgecolors='none')

def get_points(filePath):
  points = [[], []]
  for index, line in enumerate(open(filePath, 'r')):
    if 'id' in line:
      continue
    items = line.split(',')
    label = int(items[-1])
    # features = hash(','.join(items[1:-1]))
    key = index
    points[0].append(key)
    points[1].append(label)
  return points

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print '''
Usage:
  python plot.py predict_path result_path
'''
  # 读取预测数据集以及预测结果数据集
  predictData = [[], []]
  resultData = [[], []]
  for line in open(sys.argv[1], 'r'):
    items = line.split(',')
    label = int(items[-1])
    key = hash(','.join(items[1:-1]))
    predictData[0].append(key)
    resultData[0].append(key)
    predictData[1].append(label)
  for line in open(sys.argv[2], 'r'):
    if 'id' in line:
      continue
    items = line.split(',')
    label = int(items[-1])
    resultData[1].append(label)

  plot(predictData, resultData)
