
import argparse
import heapq
import logging
import os
import pickle as pkl
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm


from tensorboardX import SummaryWriter

from frcnn_eval.pascal_voc import voc_eval_kit

from model.wsddn_vgg16 import WSDDN_VGG16
from utils.wsddn_dataset import WSDDNDataset

from utils.util import make_directory, tensor_to_image, draw_box
from utils.net import adjust_learning_rate, save_checkpoint
from utils.cpu_nms import cpu_nms as nms


# Set logger
logger = logging.getLogger('DataLoader')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)


def get_args():
    parser = argparse.ArgumentParser(description='Train')
    
    parser.add_argument('--train', type=str, default='WSDDN', help='Training name')
    parser.add_argument('--session', type=int, default=2, help='Distinguish training session')
    
    # Parameter
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', help='train / test')
    
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Starting learning rate', )
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--epoch', dest='max_epoch', type=int, default=20)
    parser.add_argument('--N_classes', type=int, default=20)
    
    parser.add_argument('--min_prop', type=int, default=20, help='minimum proposal box size')
    
    # Directory
    parser.add_argument('--dataroot', default='datasets')
    parser.add_argument('--jpeg_path', default='JPEGImages')
    parser.add_argument('--text_path', default='annotations.txt')
    parser.add_argument('--ssw_path', default='voc_2007_trainval.mat')
    parser.add_argument('--image_label_path', default='voc_2007_trainval_image_label.json')
    parser.add_argument('--checkpoints_path', default='checkpoints')
    parser.add_argument('--tensorboard_path', default='tensorboard')
    
    # Test Directory
    parser.add_argument('--result_path', default='result')
    
    # Visualize
    parser.add_argument('--disp_count', type=int, default=1000, help='Number of iterations to display loss')
    parser.add_argument('--save_count', type=int, default=5, help='Number of epochs to save checkpoints')
    
    # Pretrained
    parser.add_argument('--pretrained_name', type=str, default='VGG16', help='VGG16 / Alexnet')
    parser.add_argument('--pretrained_path', type=str, default='pretrained_model', help='Load pretrained model')
    parser.add_argument('--pretrained_model', type=str, default='vgg16_caffe.pth')
    
    # Test Parameter
    parser.add_argument('--max_per_set', type=int, default=40, help='Keep an avg of 40 detection per class per images prior to NMS')
    parser.add_argument('--max_per_prop', type=int, default=100, help='Keep at most 100 detection per class per image prior to NMS')
    
    args = parser.parse_args()
    
    return args


def test(args, model, test_dataset):
    model.cuda()
    model.eval()
    
    eval_kit = voc_eval_kit('test', '2007', os.path.join(args.dataroot, 'VOCdevkit'))
    
    result_dir = os.path.join(args.result_path, str(args.session))
    start = time.time()
    
    # DataLoader
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=0)
    
    save_boxes = os.path.join(result_dir, '{}.pkl'.format(args.pretrained_name))
    
    with open(save_boxes, 'rb') as f:
        all_boxes = pkl.load(f)
        
    
    # Calculate NMS
    for c in range(args.N_classes):
        for index in range(len(test_dataset)):
            dets = all_boxes[c][index]
            if dets == []:
                continue
            
            nms_result = nms(dets, 0.4)
            
            all_boxes[c][index] = dets[nms_result, :].copy()
    for j in range(args.N_classes):
        for i in range(len(test_dataset)):
            inds = np.where(all_boxes[j][i][:] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]
            print(all_boxes)
            print('inds', inds.shape, type(inds))
    print('NMS complete, time: %.3f', time.time() - start)
    eval_kit.evaluate_detections(all_boxes)
    
    
if __name__ == '__main__':
    args = get_args()
    logger.info(args)
    
    # Setting GPU number
    torch.cuda.set_device(args.gpu_id)
    
    # Model
    model = WSDDN_VGG16(args)
    
    # Train Directory
    if args.mode == 'train':
        root_dir = args.train
        session_dir = os.path.join(root_dir, str(args.session))
        checkpoints_dir = os.path.join(session_dir, args.checkpoints_path)
        tensorboard_dir = os.path.join(session_dir, args.tensorboard_path)
        
        make_directory(root_dir, session_dir, checkpoints_dir, tensorboard_dir)
        train_dataset = WSDDNDataset(args)

        # Model Summary Writer
        board = SummaryWriter(log_dir=os.path.join(tensorboard_dir))
        
        # Train
        train(args, model, train_dataset, board, checkpoints_dir)
        
    elif args.mode == 'test':
        root_dir = args.result_path
        session_dir = os.path.join(root_dir, str(args.session))
        
        make_directory(root_dir, session_dir)
        test_dataset = WSDDNDataset(args)
        
        # Load Trained model
        trained_model = os.path.join(args.train, str(args.session), args.checkpoints_path, (args.pretrained_name + '_' + str(args.session) + '_final.pth'))
        if os.path.exists(trained_model):
            checkpoint = torch.load(os.path.join(trained_model))
            
            model.load_state_dict(checkpoint['model'])
            logger.info("Loaded checkpoint %s" %(trained_model))
        
        else:
            logger.debug("There is no model")
        
        # Test
        test(args, model, test_dataset)
