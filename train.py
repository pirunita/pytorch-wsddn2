
import argparse
import os

import torch
import torch.nn as nn
import tqdm
from datasets.wsddn_dataset import WSDDNDataset
from model.wsddn_vgg16 import WSDDN_VGG16
from utils.net import adjust_learning_rate
def get_args():
    parser = argparse.ArgumentParser(description='Train')
    
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--datamode', type=str, default='train', help='train / test')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.00001, help='Starting learning rate', )
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--epochs', dest='max_epochs', type=int, default=20)
    
    parser.add_argument('--min_prop', type=int, default=20, help='minimum proposal box size')
    
    # Directory
    parser.add_argument('--dataroot', default='data')
    parser.add_argument('--ssw_path', default='voc_2007_trainval.mat')
    parser.add_argument('--jpeg_path', default='JPEGImages')
    parser.add_argument('--text_path', default='annotations.txt')
    parser.add_argument('--image_label_path', default='voc_2007_trainval_image_label.json')
    
    
    # Model Directory
    parser.add_argument('--pretrained_dir', type=str, default='pretrained', help='Load pretrained model')
    
    
    args = parser.parse_args()
    
    return args


def train(args, model, dataset):
    
    model.train()
    
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        running_loss = 0.0
        
        for i, (file_name, image, proposals, labels) in tqdm.tqdm(enumerate(train_loade)):
            image = image.cuda()
            proposals = proposals.cuda()
            labels = labels.cuda()
        
        if epoch == 10:
            adjust_learning_rate(optimizer, 0.1)

if __name__ == '__main__':
    args = get_args()
    print(args)
    
    # Setting GPU number
    torch.cuda.set_device(args.gpu_id)
    
    # model
    model = WSDDN_VGG16(args)
    
    # Train Dataset
    train_data = WSDDNDataset(args)