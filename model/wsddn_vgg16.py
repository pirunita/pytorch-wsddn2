
import logging
import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from model.roi_align.modules.roi_align import RoIAlignAvg
from model.roi_pooling.modules.roi_pool import _RoIPooling


# Set logger
logger = logging.getLogger('DataLoader')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)


class WSDDN_VGG16(nn.Module):
    def __init__(self, args, num_classes=20):
        super(WSDDN_VGG16, self).__init__()
        
        self.args = args
        self.num_classes = num_classes
        
        self.pretrained_dir = os.path.join(args.dataroot, args.pretrained_path, args.pretrained_model)
        # VGG16
        vgg_model = torchvision.models.vgg16()
        if self.pretrained_dir is None:
            logger.debug('There is no VGG16 pretrained model')
        else:
            logger.info('Loading pretrained VGG16')
            state_dict = torch.load(self.pretrained_dir)
            vgg_model.load_state_dict({k: v for k, v in state_dict.items() if k in vgg_model.state_dict()})
        
        self.base_network = nn.Sequential(*list(vgg_model.features._modules.values())[:-1])
        self.top_network = nn.Sequential(*list(vgg_model.classifier._modules.values())[:-1])
        self.fc8c = nn.Linear(4096, self.num_classes)
        self.fc8d = nn.Linear(4096, self.num_classes)
        
        self.roi_pooling = _RoIPooling(7, 7, 1.0 / 16.0)
        self.roi_align = RoIAlignAvg(7, 7, 1.0 / 16.0)
        
        
    def forward(self, image, rois, proposal_score=None, image_level_label=None):
        N = rois.size(0)
        feature_map = self.base_network(image)
        
        zero_padded_rois = torch.cat([torch.zeros(N, 1).to(rois), rois], 1)
        pooled_feature_map = self.roi_pooling(feature_map, zero_padded_rois).view(N, -1)
        
        """        
        if proposal_score is not None:
            pooled_feature_map = pooled_feature_map * (proposal_score.view(N, 1))
        """
        
        fc7 = self.top_network(pooled_feature_map)
        fc8c = self.fc8c(fc7)
        fc8d = self.fc8d(fc7) / 2
        
        sigma_clf = F.softmax(fc8c, dim=1)
        sigma_det = F.softmax(fc8d, dim=0)
        
        scores = sigma_clf * sigma_det
        
        if image_level_label is None:
            return scores
        
        image_level_score = torch.sum(scores, 0)
        image_level_score = torch.clamp(image_level_score, min=0, max=1)
        
        loss = F.binary_cross_entropy(image_level_score, image_level_label.to(torch.float32), size_average=False)
        
        return scores, loss
        