
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
    parser.add_argument('--session', type=int, default=3, help='Distinguish training session')
    
    # Parameter
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', help='train / test')
    
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--alpha', type=float, default=0.0001, help='alpha for spatial regularization')
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
    
    parser.add_argument('--log_path', default='logs')
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


def train(args, model, train_dataset, board, log_writer, checkpoint_dir):
    np.random.seed(3)
    torch.manual_seed(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(5)

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
        reg_sum = 0
        start_time = time.time()
        
        
        for step, (file_name, image, proposals, labels) in tqdm.tqdm(enumerate(train_loader), desc='Dataset iterate'):
            image = image.cuda()
            proposals = proposals.cuda()
            labels = labels.cuda()

            # No regressor
            # image_level_label [1, 20]
            scores, loss, reg = model(image, proposals, image_level_label=labels[0])
            
            # Iterate Update
            reg = reg * args.alpha
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
                logger.info("net %s, epoch %2d, iter %4d, time: %.3f, loss: %.4f, reg: %.4f, lr: %.2e" %
                            (args.pretrained_name, epoch, step, t, loss_sum / iter_sum, reg_sum / iter_sum, lr))
                log_writer.write("[net %s][session %d][epoch %2d][iter %4d][time %.3f][loss %.4f][reg %.4f][lr %.2e]\n" %
                            (args.pretrained_name, args.session, epoch, step, t, loss_sum / iter_sum, reg_sum / iter_sum, lr))
                loss_sum = 0
                reg_sum = 0
                iter_sum = 0
        
        log_writer.flush()
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

    log_writer.write('save model: {}'.format(final_checkpoint_name))    
    log_writer.close()


def test(args, model, test_dataset):
    np.random.seed(3)
    torch.manual_seed(3)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(5)
        
    model.cuda()
    model.eval()
    
    eval_kit = voc_eval_kit('test', '2007', os.path.join(args.dataroot, 'VOCdevkit'))
    
    result_dir = os.path.join(args.result_path, str(args.session))
    start = time.time()
    
    # DataLoader
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=0)
    all_boxes = [[[] for _ in range(len(test_dataset))] for _ in range(args.N_classes)]
    
    """
    print('all_boxes', len(all_boxes))
    print('type', type(all_boxes))
    print('all_boxes[0] length', len(all_boxes[0]))
    #print('all_boxes[0]', all_boxes[0])
    print('all_boxes[0][0]', all_boxes[0][0][0])
    print('x', all_boxes[0][k])
    """
    # Set to update threshold
    max_per_set = args.max_per_set * len(test_dataset)
    max_per_prop = args.max_per_prop
    print('max_per_set', max_per_set)
    print('max_per_prop', max_per_prop)
    thresh = -np.inf * np.ones(20)
    minheap_cls_scores = [[] for _ in range(20)]
    
    
    for step, (file_name, image, proposals, labels) in tqdm.tqdm(enumerate(test_loader)):
        start_time = time.time()
        
        image = image.cuda()
        proposals = proposals.cuda()
        labels = labels.cuda()
        # scores, (# of proposals, classes)
        
        scores = model(image, proposals, image_level_label=labels).detach().cpu().numpy()
        scores = scores * 1000
        
        image = tensor_to_image(image.squeeze(0))
        
        #print('scores', scores.shape)
        
        #print('thresh', thresh)
        """
        for c in range(args.N_classes):
            indices = np.where()
        """
        for c in range(args.N_classes):
            # Indexing, (# of proposals)
            indices = np.where((scores[:, c] > thresh[c]))[0]
            #print('indices.shape', indices.shape)
            #print('indices', indices)
            
            # Scores per proposals in each class, (proposals,)
            cls_scores = scores[indices, c]
            #print('cls_scores.shape', cls_scores.shape)
            #print('cls_scores', cls_scores)
            
            # Boxes per proposals, (proposals, 4)
            cls_boxes = proposals[0][indices]
            
            #print('cls_boxes.type', type(cls_boxes))
            #print('cls_boxes.shape', cls_boxes.shape)
            #print('cls_boxes', cls_boxes)
            
            
            # Sorting highscore 
            top_indices = np.argsort(-cls_scores)[:args.max_per_prop]
            #print("top 100 proposal's indices score", top_indices)
            """
            if cls_scores.shape[0] < 100:
                print('proposal from model', cls_scores.shape)
            """
            # Top 100 scores & proposals
            try:
                top100_scores = cls_scores[top_indices]
                top100_boxes = cls_boxes[top_indices, :]
            except Exception as e:
                pass
                """
                print(e)
                print(top100_scores.shape)
                print(top100_boxes.shape)
                """
            # Push new scores in minheap
            for val in cls_scores:
                heapq.heappush(minheap_cls_scores[c], val)
            #print('minheap_cls_scores[c].len', len(minheap_cls_scores[c]))
            
            # Score Threshold Update
            if len(minheap_cls_scores[c]) > max_per_set:
                #print(print('minheap', len(minheap_cls_scores[c])))
                while len(minheap_cls_scores[c]) > max_per_set:
                    heapq.heappop(minheap_cls_scores[c])
                thresh[c] = minheap_cls_scores[c][0]
                
            try:
                all_boxes[c][step] = np.hstack((top100_boxes, top100_scores[:, np.newaxis])).astype(np.float32, copy=False)
            except Exception as e:
                
                pass
                """
                print(e)
                print(top100_scores.shape)
                print(top100_boxes.shape)
                """
        
        
        """
        if cls_scores[0] > 0.001: 
            print(cls_boxes.shape)
            keep = nms(cls_boxes.cpu().numpy(), 0.4)
            draw_box(image, cls_boxes[keep, :], result_dir, file_name[0])
        """
        if step % 100 == 99:
            print('thresh', thresh)
            print('cls_scores', cls_scores.shape, type(cls_scores))
            print('cls_boxes', cls_boxes.shape, type(cls_boxes))
            print('top_dincies', top_indices.shape, type(top_indices))
            logger.info('%d images complete, time: %.3f' %(step + 1, time.time() - start))
            print('indice', indices.shape, type(indices))
            print('thresh', thresh.shape, type(thresh))
            print('all_bex', len(all_boxes), type(all_boxes))
            print('all_bex[0]', len(all_boxes[0]), type(all_boxes[0]))
            print('score', scores.shape, type(scores)) 
    
    """
    for j in range(args.N_classes):
        for i in range(len(test_dataset)):
            inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]
            #print(all_boxes)
            print('inds', inds.shape, type(inds))
            
            #print(all_boxes.shape)
            print(len(all_boxes))
    """
    
    save_boxes = os.path.join(result_dir, '{}.pkl'.format(args.pretrained_name))
    pkl.dump(all_boxes, open(save_boxes, 'wb'))
    
    logger.info('Detection complete, elapsed time: %.3f', time.time() - start)
    
    # Calculate NMS
    for c in range(args.N_classes):
        for index in range(len(test_dataset)):
            dets = all_boxes[c][index]
            if dets == []:
                continue
            
            keep = nms(dets, 0.4)
            
            all_boxes[c][index] = dets[keep, :].copy()
    
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
        
        log_dir = os.path.join(session_dir, args.log_path)
        checkpoints_dir = os.path.join(session_dir, args.checkpoints_path)
        tensorboard_dir = os.path.join(session_dir, args.tensorboard_path)
        
        make_directory(root_dir, session_dir, checkpoints_dir, tensorboard_dir, log_dir)
        train_dataset = WSDDNDataset(args)

        # Model Summary Writer
        board = SummaryWriter(log_dir=os.path.join(tensorboard_dir))
        
        # Log training record
        log_file_dir = os.path.join(log_dir, 'log_{}_{}.txt'.format(args.pretrained_name, args.session))
        log_writer = open(log_file_dir, 'a')
        log_writer.write(str(args))
        log_writer.write('\n')
        
        
        # Train
        train(args, model, train_dataset, board, log_writer, checkpoints_dir)
        
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
