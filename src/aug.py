import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random

def realize_relativebox(box,img_shape):# 将坐标转换到真实的xyxy
    im_h,im_w = img_shape[:2]
    x,y,w,h = box.ravel()
    x = x*im_w
    y = y*im_h
    h = h*im_h
    w = w*im_w
    return np.array([x-w/2,y-h/2,x+w/2,y+h/2])

def multi_box_iou_xywh(box1, box2):
    """一一对应得算IoU
    In this case, box1 or box2 can contain multi boxes.
    Only two cases can be processed in this method:
       1, box1 and box2 have the same shape, box1.shape == box2.shape
       2, either box1 or box2 contains only one box, len(box1) == 1 or len(box2) == 1
    If the shape of box1 and box2 does not match, and both of them contain multi boxes, it will be wrong.
    """
    assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
    assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."


    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    inter_w = np.clip(inter_w, a_min=0., a_max=None)
    inter_h = np.clip(inter_h, a_min=0., a_max=None)

    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area)

def box_crop(boxes, labels, crop, img_shape):
    x, y, w, h = map(float, crop)# 非相对坐标
    im_w, im_h = map(float, img_shape)

    boxes = boxes.copy()# 将真实框转回xyxy, shape = (1,4)，且非【0，1】形式的相对坐标
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2) * im_w, (
        boxes[:, 0] + boxes[:, 2] / 2) * im_w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2) * im_h, (
        boxes[:, 1] + boxes[:, 3] / 2) * im_h

    crop_box = np.array([x-w/2,y-h/2,x+w/2,y+h/2])# 这句是官方没有的，也是官方错的原因，这样促使判断为中心是否在crop里
    #crop_box = np.array([x, y, x + w, y + h])# 裁切，shape = （4,），官方代码这样写不合理
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0# 真实框的中心
    mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(
        axis=1)# 判断真实框中心在不在裁切里， shape = (1,)

    boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])# 求交集，详细效果见b站搜目标检测的数据增强后的第一个视频的山羊效果展示
    boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
    boxes[:, :2] -= crop_box[:2]
    boxes[:, 2:] -= crop_box[:2]# 将boxes变成了boxes和crop_box的交集们，并变成相对于左上角的坐标（毕竟新图的左上角是crop的左上角为0，0）

    mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))# 判断左上角和右下角是否合理,至少要有一个像素
    boxes = boxes * np.expand_dims(mask.astype('float32'), axis=1)
    labels = labels * mask.astype('float32')# 不合条件的都变成0
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / w, (
        boxes[:, 2] - boxes[:, 0]) / w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / h, (
        boxes[:, 3] - boxes[:, 1]) / h


    return boxes, labels, mask.sum()#boxes依然是相对的，相应于crop进行了变换

# 随机改变亮暗、对比度和颜色等
def random_distort(img):
    # 随机改变亮度
    def random_brightness(img, lower = 0.9, upper = 1.5):#lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)
    # 随机改变对比度
    def random_contrast(img, lower=0.8, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)
    # 随机改变颜色
    def random_color(img, lower=0.9, upper=1.1):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)

    img = Image.fromarray(img)
    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)
    img = np.asarray(img)

    return img

# 随机填充
def random_expand(img,
                  gtboxes,
                  max_ratio=1.2,#4.,
                  fill=None,
                  keep_ratio=True,
                  prob=0.5):#0.5):#执行此操作的概率
    if random.random() > prob:
        return img, gtboxes

    if max_ratio < 1.0:
        return img, gtboxes

    h, w, c = img.shape
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x# 图片大小的放缩比例，但是字符部分等比例放缩不会变形，其他部分进行填充
    else:
        ratio_y = random.uniform(1, max_ratio)
    oh = int(h * ratio_y)
    ow = int(w * ratio_x)
    off_x = random.randint(0, ow - w)# 字符部分偏移左上角的距离
    off_y = random.randint(0, oh - h)

    out_img = np.zeros((oh, ow, c))
    if fill and len(fill) == c:# 填充成某个颜色，估计只能三色+白色，默认黑色填充
        for i in range(c):
            out_img[:, :, i] = fill[i] * 255.0

    out_img[off_y:off_y + h, off_x:off_x + w, :] = img
    gtboxes[:, 0] = ((gtboxes[:, 0] * w) + off_x) / float(ow)
    gtboxes[:, 1] = ((gtboxes[:, 1] * h) + off_y) / float(oh)
    gtboxes[:, 2] = gtboxes[:, 2] / ratio_x
    gtboxes[:, 3] = gtboxes[:, 3] / ratio_y

    return out_img.astype('uint8'), gtboxes

# 随机裁剪
def random_crop(img,# 正方形图片
                boxes,# 边框
                labels,
                scales=[0.3, 1.0],# 将在这个区间内随机抽一个数作为单边放缩系数
                max_ratio=2.0,# 限制上面的scales，免得太离谱
                constraints=None,
                max_trial=50):
    if len(boxes) == 0:
        return img, boxes

    if not constraints:# min_iou,max_iou对
        constraints = [(0.1, 1.0), (0.3, 1.0), (0.5, 1.0), (0.7, 1.0),
                       (0.9, 1.0), (0.0, 1.0)]

    img = Image.fromarray(img)
    w, h = img.size
    crops = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        for _ in range(max_trial):
            scale = random.uniform(scales[0], scales[1])
            aspect_ratio = random.uniform(max(1 / max_ratio, scale * scale), \
                                          min(max_ratio, 1 / scale / scale))# 可能缩小也可能放大
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))# 所以可能有长方形的出来
            crop_x = random.randrange(w - crop_w)
            crop_y = random.randrange(h - crop_h)
            crop_box = np.array([[(crop_x + crop_w / 2.0) / w,
                                  (crop_y + crop_h / 2.0) / h,
                                  crop_w / float(w), crop_h / float(h)]])# 转成相对的xywh【0，1】

            iou = multi_box_iou_xywh(crop_box, boxes)
            if min_iou <= iou.min() and max_iou >= iou.max():#有可能保留裁剪出很小的，也可能一个含有真框的都没有
                crops.append((crop_x, crop_y, crop_w, crop_h))# 这里存的是非相对坐标
                break

    while crops:
        crop = crops.pop(np.random.randint(0, len(crops)))
        crop_boxes, crop_labels, box_num = box_crop(boxes, labels, crop, (w, h))
        if box_num < 1:
            continue
        # img = img.crop((crop[0], crop[1], crop[0] + crop[2],
        #                 crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
        img = img.crop((crop[0]-crop[2]/2, crop[1]-crop[3]/2, crop[0] + crop[2]/2,
                        crop[1]+crop[3]/2)).resize(img.size, Image.LANCZOS)
        img = np.asarray(img)
        return img, crop_boxes, crop_labels
    img = np.asarray(img)
    return img, boxes, labels

# 随机缩放
def random_interp(img, size, interp=None):
    interp_method = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
    ]
    if not interp or interp not in interp_method:
        interp = interp_method[random.randint(0, len(interp_method) - 1)]
    h, w, _ = img.shape
    im_scale_x = size / float(w)
    im_scale_y = size / float(h)
    img = cv2.resize(
        img, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)
    return img

# 随机翻转
def random_flip(img, gtboxes, thresh=0.5):
    if random.random() > thresh:
        img = img[:, ::-1, :]
        gtboxes[:, 0] = 1.0 - gtboxes[:, 0]
    return img, gtboxes

# 随机打乱真实框排列顺序
def shuffle_gtbox(gtbox, gtlabel):
    gt = np.concatenate(
        [gtbox, gtlabel[:, np.newaxis]], axis=1)
    idx = np.arange(gt.shape[0])
    np.random.shuffle(idx)
    gt = gt[idx, :]
    return gt[:, :4], gt[:, 4]

# 图像增广方法汇总
def image_augment(img, gtboxes, gtlabels, size, means=None):# boxes是相对值
    # 随机改变亮暗、对比度和颜色等
    # img = random_distort(img)
    # 随机填充
    img, gtboxes = random_expand(img, gtboxes, fill=means)# 没有问题
    # 随机裁剪
    img, gtboxes, gtlabels, = random_crop(img, gtboxes, gtlabels)# 这一回终于修复此函数bug
    # 随机缩放
    img = random_interp(img, size)#没有问题？百分占比的好处是放缩没影响？
    # 随机翻转
    img, gtboxes = random_flip(img, gtboxes)# 也没有问题
    # 随机打乱真实框排列顺序
    gtboxes, gtlabels = shuffle_gtbox(gtboxes, gtlabels)

    # crop_show(img,gtboxes)

    return img.astype('float32'), gtboxes.astype('float32'), gtlabels.astype('int32')
#
# if __name__ == '__main__':
#     boxes = np.array([[1,2,3,1],[1,2,1,1],[2,4,5,3]])
#     centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
#     mask = np.logical_and(boxes[:2] <= centers, centers <= boxes[2:]).all(
#         axis=1)
#     mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
#     boxes = boxes * np.expand_dims(mask.astype('float32'), axis=1)
#     labels = 1 * mask.astype('float32')

def crop_show(img,boxes):
    from matplotlib import pyplot as plt
    from matplotlib import patches

    if not isinstance(img,np.ndarray):
        img = np.asarray(img)
    plt.imshow(img)
    boxes = realize_relativebox(boxes,img.shape)

    def draw_rectangle(currentAxis, bbox, edgecolor='r', facecolor='y', fill=False, linestyle='-'):
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1, linewidth=1,
                                 edgecolor=edgecolor, facecolor=facecolor, fill=fill, linestyle=linestyle)
        currentAxis.add_patch(rect)

    draw_rectangle(plt.gca(),boxes)
    plt.show()

