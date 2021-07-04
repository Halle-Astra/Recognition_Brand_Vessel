import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib.patches as patches
from PIL import Image 
import os 

def draw_det_res(dt_boxes,img):
	dt_boxes = [dt_boxes]
	if len(dt_boxes) > 0:
		import cv2
		src_im = img
		for box in dt_boxes:
			box = box.astype(np.int32).reshape((-1, 1, 2))
			cv2.polylines(src_im, [box], True, color=(255, 255), thickness=3)
	return src_im 


def get_rotate_crop_image(img, points):
	'''
	img_height, img_width = img.shape[0:2]
	left = int(np.min(points[:, 0]))
	right = int(np.max(points[:, 0]))
	top = int(np.min(points[:, 1]))
	bottom = int(np.max(points[:, 1]))
	img_crop = img[top:bottom, left:right, :].copy()
	points[:, 0] = points[:, 0] - left
	points[:, 1] = points[:, 1] - top
	'''
	img_crop_width = int(
		max(
			np.linalg.norm(points[0] - points[1]),
			np.linalg.norm(points[2] - points[3])))
	img_crop_height = int(
		max(
			np.linalg.norm(points[0] - points[3]),
			np.linalg.norm(points[1] - points[2])))
	pts_std = np.float32([[0, 0], [img_crop_width, 0],
							[img_crop_width, img_crop_height],
							[0, img_crop_height]])
	M = cv2.getPerspectiveTransform(points, pts_std)
	dst_img = cv2.warpPerspective(
		img,
		M, (img_crop_width, img_crop_height),
		borderMode=cv2.BORDER_REPLICATE,
		flags=cv2.INTER_CUBIC)
	dst_img_height, dst_img_width = dst_img.shape[0:2]
	if dst_img_height * 1.0 / dst_img_width >= 1.5:
		dst_img = np.rot90(dst_img)
	return dst_img

if __name__ == '__main__':
	def  rectify(prefix):
		f = open(r'data/detection/{}_list.txt'.format(prefix) )
		t = f.read()
		f.close()
		t = t.split('\n')
		t = [i.split('\t') for i in t if i ]
		new_root = 'data/rec/{}'.format(prefix)
		
		if not os.path.exists(new_root):
			os.makedirs(new_root)
		for i in range(len(t)):
			img_path = os.path.join('data/detection',prefix,t[i][0])
			try:
				res = np.asarray(eval(t[i][1])[0][0],dtype = np.float32)
			except:
				continue
			img = cv2.imread(img_path)
			dst_img = get_rotate_crop_image(img , points=res)
			#fig = plt.figure()
			#fig.add_subplot(121)
			#plt.imshow(draw_det_res(res,cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
			#ax = fig.add_subplot(122)
			#plt.imshow(cv2.cvtColor(dst_img,cv2.COLOR_BGR2RGB))
			img_name = os.path.split(img_path)[-1]
			new_path = os.path.join(new_root,img_name)
			img_new = Image.fromarray(dst_img )
			img_new.save(new_path)

	#rectify('train')
	#rectify('test')
		
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
	
	write_train = []
	for i in range(len(t_train)):
		img_name = t_train[i][0]
		img_name = 'train/'+img_name
		rec_label = t_train[i][1][0][1][0]
		write_train.append('\t'.join([img_name,rec_label]))
	f = open('data/rec/train_list_paddle.txt','w',encoding = 'utf-8')
	f.write('\n'.join(write_train))
	f.close()

	write_test = []
	for i in range(len(t_test)):
		img_name = t_test[i][0]
		img_name = 'test/'+img_name
		rec_label = t_test[i][1][0][1][0]
		write_test.append('\t'.join([img_name,rec_label]))
	f = open('data/rec/test_list_paddle.txt','w',encoding = 'utf-8')
	f.write('\n'.join(write_test))
	f.close()

	