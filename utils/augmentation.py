import numpy as np

import torch
import torchvision

from PIL import Image, ImageOps


class Augmentation(object):
    def __init__(self):
        self.compose = Compose([
            RandomRescale(),
            RandomHorizontalFlip(),
        ])
    
    def __call__(self, image, boxes):
        return self.compose(image, boxes)


class Compose(object):
    """composes several augmentations together
    Args:
        transforms (List[Transform]): list of trnasforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, boxes):
        for transform in self.transforms:
            image, boxes = transform(image, boxes)
        
        return image, boxes


class RandomRescale(object):
    def __init__(self):
        self.target_size_list = [480, 576, 688, 864, 1200]
    
    def __call__(self, image, boxes):
        max_target_size = np.random.choice(self.target_size_list)
        print('resize', max_target_size)
        ori_img_width, ori_img_height = image.size
        max_length = max(ori_img_width, ori_img_height)
        min_length = min(ori_img_width, ori_img_height)
        
        scale_factor = max_target_size / float(max_length)
        
        if max_length * scale_factor > 2000:
            scale_factor = 2000 / max_length
        
        min_target_size = int(min_length * scale_factor)
        
        if ori_img_width >= ori_img_height:
            rescaled_image = image.resize((max_target_size, min_target_size))
        else:
            rescaled_image = image.resize((min_target_size, max_target_size))
        
        rescaled_boxes = boxes * scale_factor
        
        return rescaled_image, rescaled_boxes
    

class RandomHorizontalFlip(object):
    def __init__(self):
        self.label = 2
    
    def __call__(self, image, boxes):
        ori_img_width, ori_img_height = image.size
        
        if np.random.randint(self.label):
            print("flip!")
            image = ImageOps.mirror(image)
            
            flipped_minX = ori_img_width - boxes[:, 2]
            flipped_maxX = ori_img_width - boxes[:, 0]
            
            boxes[:, 0] = flipped_minX
            boxes[:, 2] = flipped_maxX
        
        return image, boxes
        