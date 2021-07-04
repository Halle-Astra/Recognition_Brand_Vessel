import time
import numpy as np
import paddle
from src.yolo import YOLOv3
from src.utils import TrainDataset
import sys
import glob
import os
from matplotlib import pyplot as plt
from src.utils import  multiclass_nms
from src.utils import  draw_rectangle
from src.aug import realize_relativebox
from src.utils import  test_data_loader
import argparse

# 因此，现在作为要重新接着15轮的训练，python yolo_run.py --optimizer_new True --mode train
parser = argparse.ArgumentParser(description= 'Yolo_Param')
parser.add_argument('--optimizer_new',type = bool, default=False, help = 'get a new optimizer and not load the lateset')
parser.add_argument('--batch_size', type = int, default=2,help = 'batch size')
parser.add_argument('--mode', type = str, default='train', help = 'mode: train, valid, test, find_lr, loader_check')
parser.add_argument('--learning_rate',type = float, default=0.0000001, help = 'learning rate')
parser.add_argument('--epoch', type = int, default=15, help = 'epoch')
args = parser.parse_args()

ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]

ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

IGNORE_THRESH = .7
NUM_CLASSES = 1

VALID_THRESH = 0.01
NMS_TOPK = 400
NMS_POSK = 100
NMS_THRESH = 0.45

batch_batch = 1

def get_lr(base_lr = 0.0001, lr_decay = 0.1,last_epoch = -1):
# def get_lr(base_lr=0.0000001, lr_decay=0.01, last_epoch = -1):
    #bd = [2, 4, 8]
    bd = [2, 4, 8, 16, 18, 20]
    lr = [base_lr, base_lr * lr_decay, base_lr * lr_decay * lr_decay]
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(boundaries=bd, values=lr, last_epoch = last_epoch)
    return learning_rate

def get_last_epoch():
    if glob.glob('models/yolo_epoch*'):
        fs = glob.glob('models/yolo_epoch*')
        fs = sorted(fs,key = lambda i:int(i.split('_epoch')[-1]))
        latest_model = fs[-1]
        last_epoch = int(latest_model.split('epoch')[-1])
        return last_epoch
    else:
        return -1

def load_latest_model(model, opt, opt_new = True):
    last_epoch = get_last_epoch()
    if last_epoch>-1:
        latest_model = 'yolo_epoch'+str(last_epoch)
        model_state_dict = paddle.load(latest_model)
        model.set_state_dict(model_state_dict)
        if os.path.exists(latest_model.replace('yolo_','yolo_opt_')):
            if opt_new:
                opt_state_dict = paddle.load(latest_model.replace('yolo_','yolo_opt_'))
                opt.set_state_dict(opt_state_dict)
        return last_epoch+1
    return  0

def train():

    TRAINDIR = 'data/detection/train'
    VALIDDIR = 'data/detection/test'
    paddle.set_device(paddle.get_device())
    # 创建数据读取类
    train_dataset = TrainDataset(TRAINDIR, mode='train')
    valid_dataset = TrainDataset(VALIDDIR, mode='valid')
    # 使用paddle.io.DataLoader创建数据读取器，并设置batchsize，进程数量num_workers等参数
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=False, use_shared_memory=True)
    valid_loader = paddle.io.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, use_shared_memory=True)
    model = YOLOv3(num_classes = NUM_CLASSES)  #创建模型

    if not os.path.exists('models'):
        os.mkdir('models')
    if args.optimizer_new:
        learning_rate = get_lr(args.learning_rate)
    else:
        learning_rate = get_lr(last_epoch=get_last_epoch())
    opt = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=0.9,
        weight_decay=paddle.regularizer.L2Decay(0.0005),
        parameters=model.parameters())  # 创建优化器
    epoch_begin = load_latest_model(model,opt,args.optimizer_new)
    # opt = paddle
    # .optimizer.Adam(learning_rate=learning_rate, weight_decay=paddle.regularizer.L2Decay(0.0005), parameters=model.parameters())
    log_fname = 'train_{}.log'.format(int(time.time()))
    MAX_EPOCH = args.epoch
    for epoch in range(epoch_begin, MAX_EPOCH):
        losses = 0
        loss_cnt = 0
        for i, data in enumerate(train_loader()):
            loss_cnt += 1
            img, gt_boxes, gt_labels, img_scale = data
            gt_scores = np.ones(gt_labels.shape).astype('float32')
            gt_scores = paddle.to_tensor(gt_scores)
            img = paddle.to_tensor(img)
            gt_boxes = paddle.to_tensor(gt_boxes)
            gt_labels = paddle.to_tensor(gt_labels)
            outputs = model(img)  #前向传播，输出[P0, P1, P2]
            loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                  anchors = ANCHORS,
                                  anchor_masks = ANCHOR_MASKS,
                                  ignore_thresh=IGNORE_THRESH,
                                  use_label_smooth=False)        # 计算损失函数
            losses += loss
            if loss_cnt%batch_batch == 0:
                losses.backward()    # 反向传播计算梯度
                opt.step()  # 更新参数
                opt.clear_grad()
                losses = 0
                loss_cnt = 0
            if i % 10 == 0:
                timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                print('{}[TRAIN]epoch {}, iter {}, learning rate: {}, output loss: {}'.format(timestring, epoch, i, opt.get_lr(), loss.numpy()))
                log_f = open(log_fname,'a')
                log_f.write('{}[TRAIN]epoch {}, iter {}, learning rate: {}, output loss: {}'.format(timestring, epoch, i, opt.get_lr(), loss.numpy())+'\n')
                log_f.close()
            del outputs, loss, gt_boxes, gt_labels, gt_scores, img
        learning_rate.step()
        # save params of model
        if (epoch % 5 == 0) or (epoch == MAX_EPOCH -1):
            paddle.save(model.state_dict(), 'models/yolo_epoch{}'.format(epoch))
            paddle.save(opt.state_dict(), 'models/yolo_opt_epoch{}'.format(epoch))

        # 每个epoch结束之后在验证集上进行测试
        with paddle.no_grad():
            model.eval()
            for i, data in enumerate(valid_loader()):
                img, gt_boxes, gt_labels, img_scale = data
                gt_scores = np.ones(gt_labels.shape).astype('float32')
                gt_scores = paddle.to_tensor(gt_scores)
                img = paddle.to_tensor(img)
                gt_boxes = paddle.to_tensor(gt_boxes)
                gt_labels = paddle.to_tensor(gt_labels)
                outputs = model(img)
                loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                      anchors = ANCHORS,
                                      anchor_masks = ANCHOR_MASKS,
                                      ignore_thresh=IGNORE_THRESH,
                                      use_label_smooth=False)
                if i % 1 == 0:
                    timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                    print('{}[VALID]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy()))
                    log_f = open(log_fname,'a')
                    log_f.write('{}[VALID]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy())+'\n')
                    log_f.close()
        model.train()
        del outputs, loss, gt_boxes, gt_labels, gt_scores, img
    #log_f.close()

def valid():
    VALIDDIR = 'data/detection/test'
    paddle.set_device(paddle.get_device())
    # 创建数据读取类
    valid_dataset = TrainDataset(VALIDDIR, mode='valid')
    # 使用paddle.io.DataLoader创建数据读取器，并设置batchsize，进程数量num_workers等参数
    valid_loader = paddle.io.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                                        use_shared_memory=True)
    model = YOLOv3(num_classes=NUM_CLASSES)  # 创建模型

    epoch_begin = load_latest_model(model)
    # opt = paddle
    # .optimizer.Adam(learning_rate=learning_rate, weight_decay=paddle.regularizer.L2Decay(0.0005), parameters=model.parameters())

    # 每个epoch结束之后在验证集上进行测试
    with paddle.no_grad():
        model.eval()

        if not os.path.exists('test_result'):
            os.mkdir('test_result')
        cnt = 0
        for i, data in enumerate(valid_loader()):
            img, gt_boxes, gt_labels, img_scale = data
            img_scale = np.expand_dims(img.shape[-2:],axis = 0)
            gt_scores = np.ones(gt_labels.shape).astype('float32')
            gt_scores = paddle.to_tensor(gt_scores)
            img = paddle.to_tensor(img)
            gt_boxes = paddle.to_tensor(gt_boxes)
            gt_labels = paddle.to_tensor(gt_labels)
            outputs = model(img)
            bboxes, scores = model.get_pred(outputs,
                                            im_shape = paddle.to_tensor(img_scale),
                                            anchors = ANCHORS,
                                            anchor_masks=ANCHOR_MASKS,
                                            valid_thresh=VALID_THRESH)

            bboxes_data = bboxes.numpy()# 1, 25200, 4
            scores_data = scores.numpy()
            result = multiclass_nms(bboxes_data, scores_data,
                                    score_thresh=VALID_THRESH,
                                    nms_thresh=NMS_THRESH,
                                    pre_nms_topk=NMS_TOPK,
                                    pos_nms_topk=NMS_POSK)
            # print(outputs)
            result = result[0]
            if result != []:
                if True:#sum(result[:,0])>0:
                    mean = [0.48678351, 0.50259582, 0.53289277]
                    std = [0.28744323, 0.28601505, 0.29882203]
                    img = img.numpy()[0].transpose((1,2,0))
                    for j in range(3):
                        img[:,:,j] = img[:,:,j]*std[j]
                        img[:,:,j] = img[:,:,j]+mean[j]
                    img = (img*255).astype(np.uint8)
                    plt.imshow(img)
                    title_ls = []
                    for box_idx,bbox in enumerate(result):
                        if bbox[1] <= 0.5 or box_idx > 7:
                            continue
                        print(bbox)
                        bbox = bbox[-4:]
                        # x,y,w,h = bbox
                        # bbox = np.array([x-w/2,y-h/2,x+w/2,y+h/2])
                        #bbox = realize_relativebox(bbox, img.shape)
                        draw_rectangle(plt.gca(), bbox, edgecolor='rgbwcmyk'[box_idx])
                        title_ls.append(bbox)
                        pass
                    title_ls = [str(i) for i in title_ls]
                    title_ls = '\n'.join(title_ls)
                    plt.title(title_ls)
                    plt.savefig('test_result/{}.png'.format(i),dpi = 200 )
                    plt.close()
                    cnt+=1
            #             bbox  = bbox[-4:]
            #             draw_rectangle(plt.gca(),bbox)
            #
            #     print(result[:,:2])
            # plt.show()
            # plt.close()
            pass

def test():
    TESTDIR = 'data/detection/test'
    paddle.set_device(paddle.get_device())
    # 使用paddle.io.DataLoader创建数据读取器，并设置batchsize，进程数量num_workers等参数
    test_loader = test_data_loader(TESTDIR, batch_size= 1, mode='test')
    model = YOLOv3(num_classes=NUM_CLASSES)  # 创建模型
    load_latest_model(model)
    # 每个epoch结束之后在验证集上进行测试
    with paddle.no_grad():
        model.eval()

        if not os.path.exists('test_result'):
            os.mkdir('test_result')
        cnt = 0
        for i, data in enumerate(test_loader()):
            img_name,img, img_scale = data
            img_scale = np.expand_dims(img.shape[-2:],axis = 0)
            img = paddle.to_tensor(img)
            outputs = model(img)
            bboxes, scores = model.get_pred(outputs,
                                            im_shape = paddle.to_tensor(img_scale),# 也就是说不dataloader返回的img_scale是另外读图时传进来配合另外读图进行绘制时用的，而且loader里只能简单resize
                                            anchors = ANCHORS,
                                            anchor_masks=ANCHOR_MASKS,
                                            valid_thresh=VALID_THRESH)

            bboxes_data = bboxes.numpy()# 22743
            scores_data = scores.numpy()
            score_threshold = 0.01
            # bboxes_data = bboxes_data[0][scores_data.ravel()>score_threshold]
            # bboxes_data = np.expand_dims(bboxes_data,axis = 0)
            # scores_data = scores_data.ravel()[scores_data.ravel()>score_threshold]
            # scores_data = np.expand_dims(scores_data,axis = (0,1))
            if (scores_data.ravel()-scores_data.ravel()[0]).sum() == 0:
                continue
            result = multiclass_nms(bboxes_data, scores_data,# 如果一个都检测不到，则会有巨量的循环
                                    score_thresh=score_threshold,
                                    nms_thresh=NMS_THRESH,
                                    pre_nms_topk=NMS_TOPK,# 这个参数完全没有利用
                                    pos_nms_topk=5)#NMS_POSK)
            # print(outputs)
            result = result[0]
            if result != []:
                if True:#sum(result[:,0])>0:
                    mean = [0.48678351, 0.50259582, 0.53289277]
                    std = [0.28744323, 0.28601505, 0.29882203]
                    img = img.numpy()[0].transpose((1,2,0))
                    for j in range(3):
                        img[:,:,j] = img[:,:,j]*std[j]
                        img[:,:,j] = img[:,:,j]+mean[j]
                    img = (img*255).astype(np.uint8)
                    plt.imshow(img)
                    title_ls = []
                    for box_idx,bbox in enumerate(result):
                        if bbox[1] <= 0.5 or box_idx>7:
                            continue
                        print(bbox)
                        bbox = bbox[-4:]
                        draw_rectangle(plt.gca(), bbox, edgecolor='rgbwcmyk'[box_idx])
                        title_ls.append(bbox)
                        pass
                    title_ls = [str(i) for i in title_ls]
                    title_ls = '\n'.join(title_ls)
                    plt.title(title_ls)
                    plt.savefig('test_result/{}.png'.format(i),dpi = 200 )
                    plt.close()
                    cnt+=1


def find_lr():
    TRAINDIR = 'data/detection/train'
    paddle.set_device(paddle.get_device())
    # 创建数据读取类
    train_dataset = TrainDataset(TRAINDIR, mode='train')
    # 使用paddle.io.DataLoader创建数据读取器，并设置batchsize，进程数量num_workers等参数
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=False, use_shared_memory=True)
    model = YOLOv3(num_classes = NUM_CLASSES)  #创建模型

    load_latest_model(model)

    lr_begin = 1e-4#1e-20
    lr_end = 100# 1e-4
    lr_multi = (lr_end/lr_begin)**(1/len(train_loader))
    lr_current = lr_begin
    print(len(train_loader))

    # learning_rate = get_lr(lr_current)
    opt = paddle.optimizer.Momentum(
        learning_rate=lr_current,
        momentum=0.9,
        weight_decay=paddle.regularizer.L2Decay(0.0005),
        parameters=model.parameters())  # 创建优化器

    loss_batches = []
    lr_batches = []
    MAX_EPOCH = 1
    for epoch in range(MAX_EPOCH):
        for i, data in enumerate(train_loader()):
            lr_batches.append(lr_current)
            opt.set_lr(lr_current)
            img, gt_boxes, gt_labels, img_scale = data
            gt_scores = np.ones(gt_labels.shape).astype('float32')
            gt_scores = paddle.to_tensor(gt_scores)
            img = paddle.to_tensor(img)
            gt_boxes = paddle.to_tensor(gt_boxes)
            gt_labels = paddle.to_tensor(gt_labels)
            outputs = model(img)  #前向传播，输出[P0, P1, P2]
            loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                  anchors = ANCHORS,
                                  anchor_masks = ANCHOR_MASKS,
                                  ignore_thresh=IGNORE_THRESH,
                                  use_label_smooth=False)        # 计算损失函数
            loss_batches.append(loss)
            loss.backward()    # 反向传播计算梯度
            opt.step()  # 更新参数
            opt.clear_grad()
            lr_current = lr_current*lr_multi
            if i % 10 == 0:
                timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                print('{}[TRAIN]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy()))
        lr_batches = np.array(lr_batches).ravel()
        loss_batches = np.array(loss_batches).ravel()
        plt.plot(lr_batches, loss_batches)
        plt.show()

def loader_check():
    VALIDDIR = 'data/detection/test'
    paddle.set_device(paddle.get_device())
    # 创建数据读取类
    valid_dataset = TrainDataset(VALIDDIR, mode='valid')
    # 使用paddle.io.DataLoader创建数据读取器，并设置batchsize，进程数量num_workers等参数
    valid_loader = paddle.io.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                                        use_shared_memory=True)

    for i, data in enumerate(valid_loader()):
        print(i)
        img, gt_boxes, gt_labels, img_scale = data
        img, gt_boxes, gt_labels = img.numpy(), gt_boxes.numpy(), gt_labels.numpy()
        gt_scores = np.ones(gt_labels.shape).astype('float32')
        # gt_scores = paddle.to_tensor(gt_scores)
        img = img[0].transpose((1, 2, 0))
        w,h = img.shape[:2]
        print(img.shape)
        if gt_boxes.size!=0:# gt_boxes 是 xywh的相对坐标形式
            if True:  # sum(result[:,0])>0:
                plt.imshow(img)
                for bbox in gt_boxes:
                    bbox = bbox[0][-4:]
                    # bbox = bbox*w
                    bbox = realize_relativebox(bbox,img.shape)
                    draw_rectangle(plt.gca(), bbox)
                plt.show()

            print(gt_labels)
        plt.close()

if __name__ == '__main__':
    #if len(sys.argv)==1:
    #    sys.argv.append('test')
    #mode = sys.argv[-1]
    mode = args.mode
    print(sys.argv)
    if mode == 'train':
        train()
    elif mode == 'test':
        test()
    elif mode == 'find_lr':
        find_lr()
    elif mode == 'loader_check':
        loader_check()
    elif mode == 'valid':
        valid()
    else:
        print('Error mode!')
