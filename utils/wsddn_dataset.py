
import json
import logging
import os

import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
import torchvision

from PIL import Image
from utils.augmentation import Augmentation

# Set logger
logger = logging.getLogger('DataLoader')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

class WSDDNDataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.root = args.dataroot
        self.mode = args.mode
        
        self.ssw_path = args.ssw_path
        self.jpeg_path = args.jpeg_path
        self.text_path = args.text_path
        self.image_label_path = args.image_label_path
        
        self.min_prop_scale = args.min_prop
        #self.means = [102.9801, 115.9465, 122.7717]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Augmentation
        self.augmentation = Augmentation()
        self.transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # imdb initialize
        self.imdb = []
    
        # JSON, trainval 5011
        with open(os.path.join(self.root, self.mode, self.image_label_path), 'r') as jr:
            self.image_label_list = json.load(jr)
        
        # Proposal
        self.proposal_list = sio.loadmat(os.path.join(self.root, self.mode, self.ssw_path))['boxes'][0]
        
        # Text
        with open(os.path.join(self.root, self.mode, self.text_path), 'r') as text_f:
            for idx, file_name in enumerate(text_f.readlines()):
                file_name = file_name.rstrip()
                
                # image label parsing
                image_label_current = [0 for i in range(20)]
                image_label_list = self.image_label_list[file_name]
                
                for i in image_label_list:
                    image_label_current[i] = 1
                #print('image_label_current', image_label_current)
                preprocessed_proposals = self.preprocess_proposal(file_name, self.proposal_list[idx])
                self.imdb.append([file_name, preprocessed_proposals, image_label_current])
    
    def __getitem__(self, index):
        # Load Data
        file_name = self.imdb[index][0]
        proposals = self.imdb[index][1]
        img_label = self.imdb[index][2]
        current_img = Image.open(os.path.join(self.root, self.mode, self.jpeg_path, file_name + '.jpg'))
        
        # Augmentation
        if self.args.mode == 'train':
            image, proposals = self.augmentation(current_img, proposals)
            
        elif self.args.mode == 'test':
            image = current_img
    
        # casting
        
        image = self.transformation(image)
        proposals = torch.tensor(proposals, dtype=torch.float32)
        img_label = torch.tensor(img_label, dtype=torch.uint8)
        
        return file_name, image, proposals, img_label
        
    def __len__(self): 
        return len(self.imdb) 
    
    
    #def get_data(self, index, h_flip=False, target_image_size=688): 
        
    def preprocess_proposal(self, file_name, boxes):
        proposals = []
        # 0: minY, 1: minX, 2: maxY, 3: maxX
        
        # Reindexing Boxes
        boxes = boxes.astype(np.float) - 1
        
        scores = np.zeros(len(boxes))
        preprocessed_index = (boxes[:, 2] >= boxes[:, 0] + self.min_prop_scale) * (boxes[:, 3] >= boxes[:, 1] + self.min_prop_scale)
        preprocessed_index = np.nonzero(preprocessed_index)[0]
        preprocessed_proposal = boxes[preprocessed_index]
        #scores = scores[preprocessed_index]
        
        # minX, minY, maxX, maxY
        proposals = np.concatenate([preprocessed_proposal[:, 1:2], preprocessed_proposal[:, 0:1], \
                                               preprocessed_proposal[:, 3:4], preprocessed_proposal[:, 2:3]], 1)
        
        return proposals
