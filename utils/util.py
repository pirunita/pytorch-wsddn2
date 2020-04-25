import os

import torch

from PIL import Image, ImageDraw
from utils.cpu_nms import cpu_nms as nms
"""
def draw_box(boxes, c='black'):
    # minX, minY, maxX, maxY
    for _, (minX, minY, maxX, maxY) in enumerate(boxes):
        plt.hlines(minY, minX, maxX, colors=c, lw=2)
        plt.hlines(maxY, minX, maxX, colors=c, lw=2)
        plt.vlines(minX, minY, maxY, colors=c, lw=2)
        plt.vlines(maxX, minY, maxY, colors=c, lw=2)
"""


def all_pair_iou(boxes_a, boxes_b):
    """
    Compute the IoU of all pairs
    :param boxes_a: (n, 4) minmax form boxes
    :param boxes_b: (m, 4) minmax form boxes
    :return: (n, m) iou of all pairs of two set
    """

    N = boxes_a.size(0)
    M = boxes_b.size(0)
    max_xy = torch.min(boxes_a[:, 2:].unsqueeze(1).expand(N, M, 2), boxes_b[:, 2:].unsqueeze(0).expand(N, M, 2))
    min_xy = torch.max(boxes_a[:, :2].unsqueeze(1).expand(N, M, 2), boxes_b[:, :2].unsqueeze(0).expand(N, M, 2))
    inter_wh = torch.clamp((max_xy - min_xy + 1), min=0)
    I = inter_wh[:, :, 0] * inter_wh[:, :, 1]
    A = ((boxes_a[:, 2] - boxes_a[:, 0] + 1) * (boxes_a[:, 3] - boxes_a[:, 1] + 1)).unsqueeze(1).expand_as(I)
    B = ((boxes_b[:, 2] - boxes_b[:, 0] + 1) * (boxes_b[:, 3] - boxes_b[:, 1] + 1)).unsqueeze(0).expand_as(I)
    U = A + B - I
    return I / U


def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by 솓
    test_net method.
    """
    num_classes = len(all_boxes)
    num_image = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_image)] for _ in range(num_classes)]
    
    for cls_ind in range(num_classes):
        for im_ind in range(num_image):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
        
    return nms_boxes
    

def draw_box(image, boxes, result_dir, file_name):
    
    rect_image = ImageDraw.Draw(image)
    for box in boxes:
        rect_image.rectangle([(box[0], box[1]), (box[2], box[3])], fill=None, outline='red', width=2)
    
    image.save(os.path.join(result_dir, str(file_name) + '.png'))
def tensor_to_image(img_tensors):
    tensor = (img_tensors.clone()+1)*0.5 * 255
    tensor = tensor.cpu().clamp(0, 255)
    
    array = tensor.numpy().astype('uint8')
    if array.shape[0] == 1:
        array = array.squeeze(0)
    elif array.shape[0] == 3:
        array = array.swapaxes(0, 1).swapaxes(1, 2)
    
    return Image.fromarray(array)


def make_directory(*dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
            
    
