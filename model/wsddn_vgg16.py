
import logging
import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from utils.util import all_pair_iou
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
        # VGG16, pth(weakly게 아닐 수도 있음)
        vgg_model = torchvision.models.vgg16()
        # 찍어보자
        if self.pretrained_dir is None:
            logger.debug('There is no VGG16 pretrained model')
        else:
            logger.info('Loading pretrained VGG16')
            state_dict = torch.load(self.pretrained_dir)
            vgg_model.load_state_dict({k: v for k, v in state_dict.items() if k in vgg_model.state_dict()})
            """
            Network debug
            """
            for k in vgg_model.state_dict():
                print('k, v', k)
        #fc6 어디감?
        self.base_network = nn.Sequential(*list(vgg_model.features._modules.values())[:-1])
        self.top_network = nn.Sequential(*list(vgg_model.classifier._modules.values())[:-1])
        self.fc8c = nn.Linear(4096, self.num_classes)
        self.fc8d = nn.Linear(4096, self.num_classes)
        
        # OICR 참고해보자
        self.roi_pooling = _RoIPooling(7, 7, 1.0 / 16.0)
        self.roi_align = RoIAlignAvg(7, 7, 1.0 / 16.0)
        self._init_weights()

    # ? xavier ㄱㄱ
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.fc8c, 0, 0.01, False)
        normal_init(self.fc8d, 0, 0.01, False)
        
    def forward(self, image, rois, proposal_score=None, image_level_label=None):
        N = rois.size(1)
        feature_map = self.base_network(image)
        # Assume Batchsize 1
        zero_padded_rois = torch.cat([torch.zeros(N, 1).to(rois[0]), rois[0]], 1)
        pooled_feature_map = self.roi_pooling(feature_map, zero_padded_rois).view(N, -1)
        
        """        
        if proposal_score is not None:
            pooled_feature_map = pooled_feature_map * (proposal_score.view(N, 1))
        """
        
        fc7 = self.top_network(pooled_feature_map)
        fc8c = self.fc8c(fc7)
        
        # remove /2 
        fc8d = self.fc8d(fc7)
        
        sigma_clf = F.softmax(fc8c, dim=1)
        sigma_det = F.softmax(fc8d, dim=0)
        
        scores = sigma_clf * sigma_det
        
        if image_level_label is None:
            return scores
        
        if self.args.mode == 'test':
            return scores
        
        # ?
        image_level_scores = torch.sum(scores, 0)
        image_level_scores = torch.clamp(image_level_scores, min=0, max=1)
        loss = F.binary_cross_entropy(image_level_scores, image_level_label.to(torch.float32), size_average=False)
        reg = self.spatial_regulariser(rois[0], fc7, scores, image_level_label)
        
        return scores, loss, reg

    
    def spatial_regulariser(self, rois, fc7, scores, image_level_label):
        K = 10
        th = 0.6
        N = rois.size(0)
        ret = 0
        for c in range(self.num_classes):
            if image_level_label[c].item() == 0:
                continue
            topk_scores, topk_indices = scores[:, c].topk(K, dim=0) 
            topk_boxes = rois[topk_indices]
            topk_features = fc7[topk_indices]

            mask = all_pair_iou(topk_boxes[0:1, :], topk_boxes).view(K).gt(th).float()

            diff = topk_features - topk_features[0]
            diff = diff * topk_scores.detach().view(K, 1)

            ret = (torch.pow(diff, 2).sum(1) * mask).sum() * 0.5 + ret
        
        return ret









