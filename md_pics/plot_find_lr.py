from matplotlib import pyplot as plt 
import numpy as np 

f = open('find_lr_new.txt')
t = f.read()
t = t.split('\n')
t = [i for i in t if i]
losses = []
for line in t:
	if 'nan' not in line and 'loss:' in line :
		loss = line.split('loss:')[-1].strip()
		loss = eval(loss)
		loss = loss[0]
		losses.append(loss)


lr_begin = 1e-4
lr_end = 100
batch_num = 1240
multi = (lr_end/lr_begin)**(1/batch_num)
lrs = [] 
lr = lr_begin  
for i in range(len(losses)):
	lrs.append(lr)
	lr*=multi 

plt.plot(lrs,losses)
plt.ylim(10,1800)
plt.show()

# @staticmethod