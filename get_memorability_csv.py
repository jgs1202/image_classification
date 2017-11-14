#coding:utf-8

import csv   #csvモジュールをインポートする
import numpy as np


g = open('targets393_metadata.csv', 'rt')
dataReader = csv.reader(g)
data = [ e for e in dataReader]
g.close()
length = len(data)
List = np.zeros( (length-1, 3), dtype=object)

for i in range(length-1):
	List[i][0] = str(data[i+1][0])
	List[i][1] = float(data[i+1][-4])
	List[i][2] = float(data[i+1][-6])
	# print(i)

train=List[:-20]
test=List[-20:]


f = open('train_metadata.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerows(train.tolist())
f.close()
h = open('test_metadata.csv', 'w')
writer = csv.writer(h, lineterminator='\n')
writer.writerows(test.tolist())
h.close()