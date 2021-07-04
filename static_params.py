from matplotlib import pyplot as plt 
import numpy as np
import os
from PIL import Image

def get_list():
	f_train = open('data/detection/train_list.txt')
	t_train = f_train.read()
	f_train.close()
	f_test = open('data/detection/test_list.txt')
	t_test = f_test.read()
	f_test.close()
	t_train = t_train.split('\n')
	t_train = [i.split('\t') for i in t_train]
	t_train = [[i[0], eval(i[1])] for i in t_train if i]
	t_test = t_test.split('\n')
	t_test = [i.split('\t') for i in t_test]
	t_test = [[i[0], eval(i[1])] for i in t_test if i]
	return t_train,t_test

def plot_reliability():
	plt.rcParams['font.sans-serif'] = ['SimHei']
	def get_reliability(prefix):
		f = open(f'data/detection/{prefix}_list.txt')
		t = f.read()
		t = t.split('\n')
		t = [i.split('\t') for i in t if i ]
		t = [eval(i[1])[0][1] for i in t]
		t = [i[1] for i in t]
		return t

	rl = get_reliability('train')+get_reliability('test')
	rl = np.array(rl)
	plt.figure(figsize = (10,5))
	plt.hist(rl,20)
	plt.savefig('md_pics/reliability.png')
	plt.show()

def plot_act():
	x = np.arange(-5,5,0.1)
	sigmoid = 1/(1+np.exp(-x))
	relu = np.maximum(x,0)
	tanh = np.tanh(x)
	fig, ax = plt.subplots(figsize = (8,5))
	plt.plot(x,sigmoid)
	plt.plot(x,tanh)
	plt.plot(x,relu)

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_color('none')
	ax.xaxis.set_ticks_position('bottom')
	ax.spines['bottom'].set_position(('data', 0))
	ax.yaxis.set_ticks_position('left')
	ax.spines['left'].set_position(('data', 0))
	plt.ylim(-1.1,1.1)
	plt.yticks([-1,-0.5,0.5,1])
	plt.legend(['sigmoid','tanh','ReLU'])
	# plt.box('off')
	plt.savefig('md_pics/act.png',dpi = 100)
	plt.show()

def plot_wh_scatter():
	fs_train = os.listdir('data/detection/train')
	fs_train = [os.path.join('data/detection/train',i) for i in fs_train]
	fs_test = os.listdir('data/detection/test')
	fs_test = [os.path.join('data/detection/test',i) for i in fs_test]
	fs = fs_train+fs_test
	wh_ls = []
	for img_path in fs:
		img = Image.open(img_path)
		wh_ls.append(img.size)
	plt.figure(figsize = (8,4))
	wh_ls = np.array(wh_ls)
	plt.plot(wh_ls[:,0],wh_ls[:,1],'.',linewidth = 0.5,alpha = 0.5)
	plt.xlabel('width')
	plt.ylabel('height')
	plt.savefig('md_pics/wh_distribution.png',dpi = 200)
	plt.show()

def stat_anchors():
	train_root = 'data/detection/train'
	test_root = 'data/detection/test'
	f_train = open('data/detection/train_list.txt')
	t_train = f_train.read()
	f_train.close()
	f_test = open('data/detection/test_list.txt')
	t_test = f_test.read()
	f_test.close()
	t_train = t_train.split('\n')
	t_train = [i.split('\t') for i in t_train ]
	t_train = [[i[0],eval(i[1])] for i in t_train if i  ] 
	t_test = t_test.split('\n')
	t_test = [i.split('\t') for i in t_test ]
	t_test = [[i[0],eval(i[1])] for i in t_test if i ] 
	gt_bbox = np.zeros((1,4))

	ratio_w_train = [] 
	ratio_h_train = []
	for i in t_train:
		img_name = i[0]
		obj = i[1][0]
		bbox_four_points,(_,_) = obj
		gt_bbox[0,:2] = np.min(bbox_four_points, axis = 0)
		gt_bbox[0,2:] = np.max(bbox_four_points, axis = 0)
		img_w,img_h = Image.open(os.path.join('data/detection/train',img_name)).size
		bbox_w = gt_bbox[0,2]-gt_bbox[0,0]
		bbox_h = gt_bbox[0,3]-gt_bbox[0,1]
		ratio_w = bbox_w/img_w
		ratio_h = bbox_h/img_h
		ratio_w_train.append(ratio_w)
		ratio_h_train.append(ratio_h)

	ratio_w_test,ratio_h_test = [],[]
	for i in t_test:
		img_name = i[0]
		obj = i[1][0]
		bbox_four_points,(_,_) = obj
		gt_bbox[0,:2] = np.min(bbox_four_points, axis = 0)
		gt_bbox[0,2:] = np.max(bbox_four_points, axis = 0)
		img_w,img_h = Image.open(os.path.join('data/detection/test',img_name)).size
		bbox_w = gt_bbox[0,2]-gt_bbox[0,0]
		bbox_h = gt_bbox[0,3]-gt_bbox[0,1]
		ratio_w = bbox_w/img_w
		ratio_h = bbox_h/img_h
		ratio_w_test.append(ratio_w)
		ratio_h_test.append(ratio_h)

	plt.figure(figsize = (8,5))
	plt.plot(ratio_w_train,ratio_h_train,'.',alpha = 0.5,label = 'train')
	plt.legend()
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.savefig('md_pics/ratio_train.png',dpi = 200 )
	plt.xlabel('width')
	plt.ylabel('height')
	
	plt.close()
	plt.figure(figsize = (8,5))
	plt.plot(ratio_w_test,ratio_h_test,'.',alpha = 0.5,label = 'test')
	plt.legend()
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.xlabel('relative width')
	plt.ylabel('relative height')
	plt.savefig('md_pics/ratio_test.png',dpi = 200)
	
	plt.close()
	#plt.show()

	plt.figure(figsize = (8,5))
	plt.hist(ratio_w_train,20,label = 'train',alpha = 0.5)
	plt.hist(ratio_w_test,20,label = 'test',alpha = 0.5)
	plt.xlabel('relative width')
	plt.legend()
	plt.savefig('md_pics/hist_width.png',dpi = 200 )
	plt.close()
	
	plt.figure(figsize = (8,5))
	plt.hist(ratio_h_train,20,label = 'train',alpha = 0.5)
	plt.hist(ratio_h_test,20,label = 'test',alpha = 0.5)
	plt.xlabel('relative height')
	plt.savefig('md_pics/hist_height.png',dpi = 200 )
	plt.close()


	from sklearn.cluster import KMeans
	okmeans = KMeans(n_clusters = 3)

	w_all = ratio_w_train+ratio_w_test
	h_all = ratio_h_train+ratio_h_test
	w_all = np.array(w_all)
	h_all = np.array(h_all) 
	wh_all = np.c_[w_all,h_all ]
	okmeans.fit(wh_all)
	print(okmeans.cluster_centers_)

def stat_charac():
	plt.rcParams['font.sans-serif'] = ['SimHei']
	f_train = open('data/detection/train_list.txt')
	t_train = f_train.read()
	f_train.close()
	f_test = open('data/detection/test_list.txt')
	t_test = f_test.read()
	f_test.close()
	t_train = t_train.split('\n')
	t_train = [i.split('\t') for i in t_train ]
	t_train = [[i[0],eval(i[1])] for i in t_train if i  ] 
	t_test = t_test.split('\n')
	t_test = [i.split('\t') for i in t_test ]
	t_test = [[i[0],eval(i[1])] for i in t_test if i ] 
	t_all = t_train+t_test 
	charac_all = ''
	for line in t_all:
		line = line[1][0][1][0]
		charac_all+=line
	charac_all = [i for i in charac_all if i not in list(map(str,range(10)))]
	charac_set = set(charac_all)
	dict_charac = {}
	for i in charac_set:
		dict_charac[i] = charac_all.count(i)
	
	plt.figure(figsize = (8,5))
	plt.hist(charac_all,len(charac_set),align = 'left')
	plt.xticks([])
	plt.savefig('md_pics/charac_hist_all.png',dpi = 200)
	plt.close()

	plt.figure(figsize = (8,5))
	plt.hist(charac_all,len(charac_set),align = 'left')
	plt.xlim(-2,40)
	plt.savefig('md_pics/charac_hist_50.png',dpi = 200 )
	plt.close()

	if not os.path.exists('data/rec'):
		os.makedirs('data/rec')
	f = open('data/rec/charac_dict.txt','w',encoding = 'utf-8')
	f.write('\n'.join(list(charac_set)+[str(i) for  i in range(10)]))
	f.close()

def stat_mean():
	t_train,t_test = get_list()
	t_train = [os.path.join('data/detection/train',i[0]) for i in t_train]
	t_test = [os.path.join('data/detection/test',i[0]) for i in t_test]
	pix_num = 0
	x_sum = np.zeros((3,),dtype = np.float)
	x2_sum = np.zeros((3,),dtype = np.float)
	for img_path in t_train+t_test:
		img = plt.imread(img_path)/255
		pix_num += img[:,:,0].size
		x_sum += img.sum(axis = (0,1))
		x2_sum += (img**2).sum(axis = (0,1))
	x_mean = x_sum/pix_num
	x2_mean = x2_sum/pix_num
	std_data = (x2_mean-x_mean**2)**0.5
	print(x_mean)
	print(std_data)
	# mean_all = np.array([0,0,0],dtype = np.float)
	# std_all = np.array([0,0,0],dtype = np.float)
	# sum_allpix = 0
	# for img_path in t_train+t_test:
	# 	img = plt.imread(img_path)
	# 	sum_allpix += img.sum(axis = (0,1))
	# 	# img_mean = img.mean(axis = (0,1))
	# 	# img_std = img.std(axis = (0,1))
	# 	# mean_all += img_mean
	# 	# std_all += img_std
	# # mean_all = mean_all/len(t_train+t_test)
	# print('RGB mean',mean_all)


if __name__=='__main__':
	stat_mean()