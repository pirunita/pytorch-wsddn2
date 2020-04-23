
import argparse
import logging
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm


from tensorboardX import SummaryWriter

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
    parser.add_argument('--max_per_prop', type=int, default=100)
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
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) 
    
    total_sum = 0
    for epoch in tqdm.tqdm(range(args.start_epoch, args.max_epoch + 1), desc='Training'):
        loss_sum = 0
        iter_sum = 0
        start_time = time.time()
        
        
        for step, (file_name, image, proposals, labels) in tqdm.tqdm(enumerate(train_loader), desc='Dataset iterate'):
            image = image.cuda()
            proposals = proposals.cuda()
            labels = labels.cuda()

            # No regressor
            scores, loss = model(image, proposals, image_level_label=labels)
            
            # Iterate Update
            loss_sum += loss.item()
            iter_sum += 1
            total_sum += 1
            
            # Parameter update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Visualization
            if step % args.disp_count == 0 and step > 0:
                board.add_scalar('Train/loss', loss_sum / iter_sum, total_sum)
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
    final_checkpoint_name = os.path.join(checkpoint_dir, '{}_{}_{}.pth'.format(args.pretrained_name, args.session, 'final'))
    checkpoint = dict()
    checkpoint['net'] = args.pretrained_name
    checkpoint['session'] = args.session
    checkpoint['epoch'] = epoch
    checkpoint['model'] = model.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    save_checkpoint(checkpoint, final_checkpoint_name)
    
    logger.info('save model: {}'.format(final_checkpoint_name))
    logger.info('Training Finished..')


def test(args, model, test_dataset):
    model.cuda()
    model.eval()
    
    result_dir = os.path.join(args.result_path, str(args.session))
    # DataLoader
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=0)
    
    
    for step, (file_name, image, proposals, labels) in tqdm.tqdm(enumerate(test_loader)):
        start_time = time.time()
        
        image = image.cuda()
        proposals = proposals.cuda()
        labels = labels.cuda()
        
        # scores, (proposals, class)
        scores = model(image, proposals, image_level_label=labels)
        image = tensor_to_image(image.squeeze(0))
        print('scores', scores.shape)
        for c in range(args.N_classes):
            indices = np.where((scores[:, c] > 0))[0]
            cls_scores = scores[indices, c].detach()
            cls_boxes = proposals[0][indices]
            top_indices = np.argsort(-cls_scores)[:args.max_per_prop]
            
            # targeting
            cls_scores = cls_scores[top_indices]
            cls_boxes = cls_boxes[top_indices, :]
            print('scores', cls_scores.shape)
            if cls_scores[0] > 0.001: 
                print(cls_boxes.shape)
                keep = nms(cls_boxes.cpu().numpy(), 0.4)
                draw_box(image, cls_boxes[keep, :], result_dir, file_name[0])
            
            
            
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
