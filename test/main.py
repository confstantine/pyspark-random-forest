# coding:utf-8
import sys
import random
import os
import time
import datetime

TRAIN_DATA_PERCENT = 0.7
PREDICT_DATA_PERCENT = 0.3

# 从数据集中无放回的抽取一定比例数据
def get_sub_set(data, per):
  return random.sample(data, int(len(data) * per))

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print '''
Usage:
  python main.py false
  python main.py true data_path
'''
    exit(0)

  # 根据当前时间戳生成用于保存训练数据集和测试数据集两个文件的名字
  timestamp = ''#str(time.mktime(datetime.datetime.now().timetuple()))
  if (sys.argv[1] == 'true'):
    dataset = []
    lines = open(sys.argv[2], 'r').readlines()[1:]
    trainDataset = get_sub_set(lines, TRAIN_DATA_PERCENT)
    predictDataset = get_sub_set(lines, PREDICT_DATA_PERCENT)
    trainFilePath = timestamp + 'train.csv'
    predictFilePath = timestamp + 'predict.csv'
    # 分别打开两个文件并把数据写入
    f = open(trainFilePath, 'w')
    for line in trainDataset:
      f.write(line)
    f.close()
    f = open(predictFilePath, 'w')
    for line in predictDataset:
      f.write(line)
    f.close()
    print '分割结束!'
    exit(0)

  predictDataset = open(timestamp + 'predict.csv', 'r').readlines()
  resultFilePath = timestamp + 'result.csv'
  # 读取预测结果进来
  right = 0
  predictResult = open(resultFilePath, 'r').readlines()[1:]
  for count, line in enumerate(predictResult):
    predictLabel = line.split(',')[-1].strip()
    realLabel = predictDataset[count].split(',')[-1].strip()
    if predictLabel == realLabel:
      right += 1
  # 输出正确率
  print '正确率为：' + str(float(right) / float(len(predictResult)) * 100) + '%'
  # 删除临时文件
  # os.system('rm %s' % resultFilePath)
  
