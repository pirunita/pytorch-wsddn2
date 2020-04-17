
import argparse
import logging
import os
import time

import torch
import torch.nn as nn
import tqdm

from tensorboardX import SummaryWriter

from datasets.wsddn_dataset import WSDDNDataset
from model.wsddn_vgg16 import WSDDN_VGG16
from utils.util import make_directory
from utils.net import adjust_learning_rate, save_checkpoint


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
    parser.add_argument('--session', type=int, default=1, help='Distinguish training session')
    
    # Parameter
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--datamode', type=str, default='train', help='train / test')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Starting learning rate', )
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--epochs', dest='max_epochs', type=int, default=20)
    
    parser.add_argument('--min_prop', type=int, default=20, help='minimum proposal box size')
    
    # Directory
    parser.add_argument('--dataroot', default='dataset')
    parser.add_argument('--ssw_path', default='voc_2007_trainval.mat')
    parser.add_argument('--jpeg_path', default='JPEGImages')
    parser.add_argument('--text_path', default='annotations.txt')
    parser.add_argument('--image_label_path', default='voc_2007_trainval_image_label.json')
    parser.add_argument('--checkpoints_path', default='checkpoints')
    parser.add_argument('--tensorboard_path', default='tensorboard')
    
    # Visualize
    parser.add_argument('--disp_count', type=int, default=1000, help='Number of iterations to display loss')
    parser.add_argument('--save_count', type=int, default=5, help='Number of epochs to save checkpoints')
    
    # Pretrained
    parser.add_argument('--pretrained_name', type=str, default='VGG16', help='VGG16 / Alexnet')
    parser.add_argument('--pretrained_path', type=str, default='pretrained_model', help='Load pretrained model')
    parser.add_argument('--pretrained_model', type=str, default='vgg16_caffe.pth')
    
    
    args = parser.parse_args()
    
    return args


def train(args, model, train_dataset, board, checkpoint_dir):
    model.cuda()
    model.train()
    lr = args.learning_rate
    
    # Setting Optimizer
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]
                
    optimizer = torch.optim.SGD(params, momentum=0.9)
    
    
    for epoch in tqdm.tqdm(range(args.start_epoch, args.max_epochs + 1), desc='Training'):
        loss_sum = 0
        iter_sum = 0
        start_time = time.time()
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        for step, (file_name, image, proposals, labels) in tqdm.tqdm(enumerate(train_loader), desc='Dataset iterate'):
            file_name = file_name.cuda()
            image = image.cuda()
            proposals = proposals.cuda()
            labels = labels.cuda()
        
        

            # No regressor
            scores, loss = model(image, proposals, image_level_label=labels)
            
            # Iterate Update
            running_loss += loss.item()
            iter_sum += 1
            
            # Parameter update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Visualization
            if step % args.disp_count == 0:
                board.add_scalar('Train/loss', loss_sum / iter_sum)
                t = time.time() - start_time
                logger.info("net %s, epoch %2d, iter %4d, time: %.3f, loss: %.4f, lr: %.2e" %
                            (args.pretrained_name, epoch, step, t, loss_sum / iter_sum, lr))
                loss_sum = 0
                iter_sum = 0
                
        # Decay learning rate
        if epoch == 10:
            adjust_learning_rate(optimizer, 0.1)
            lr *= 0.1
            
        if epoch % args.save_count == 0:
            checkpoint_name = os.path.join(checkpoint_dir, '{}_{}_{}.pth'.format(args.pretrained_name, args.session, epoch))
            checkpoint = dict()
            checkpoint['net'] = args.pretrained_name
            checkpoint['session'] = args.session
            checkpoint['epoch'] = epoch
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            
            save_checkpoint(checkpoint, checkpoint_name)
            logger.info('save model: {}'.format(checkpoint_name))
    
    board.close()
    save_checkpoint(checkpoint, '{}_{}_{}.pth'.format(args.pretrained_name, args.session, 'final'))
    
    
if __name__ == '__main__':
    args = get_args()
    logger.info(args)
    
    # Setting GPU number
    torch.cuda.set_device(args.gpu_id)
    
    # Result Directory
    root_dir = args.train
    session_dir = os.path.join(root_dir, args.session)
    checkpoints_dir = os.path.join(session_dir, args.checkpoints_path)
    tensorboard_dir = os.path.join(session_dir, args.tensorboard_path)
    make_directory(root_dir, session_dir, checkpoints_dir, tensorboard_dir)
    
    # Model
    board = SummaryWriter(log_dir=os.path.join(tensorboard_dir))
    model = WSDDN_VGG16(args)
    
    # Train Dataset
    train_dataset = WSDDNDataset(args)
    
    # Train
    train(args, model, train_dataset, board, checkpoints_dir)