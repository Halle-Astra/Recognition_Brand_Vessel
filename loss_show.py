import numpy as np 
from matplotlib import pyplot as plt 
import sys 

if __name__=='__main__':
	while True:
		if len(sys.argv)==1:
			sys.argv.append(r'D:\Other_All\Application_Documents\Medium\For_Work\Python\FromGit\PaddleOCR-release-2.0\output\rec/ic15/train.log')
		f = open(sys.argv[-1])
		t = f.read()
		f.close()
		t = t.split('\n')
		t = [i for i in t if ', loss:' in i ]
		t = [eval(i.split('loss:')[-1].split(',')[0].strip()) for i in t]
		data = np.array(t).ravel()
		iter = np.arange(1,(data.size)*10+1,10)
		plt.plot(iter,data)
		plt.pause(30)