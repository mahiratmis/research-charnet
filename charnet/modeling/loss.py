import torch
from torch import nn
import numpy as np


def get_dice_loss(gt_score, pred_score):
    inter = torch.sum(gt_score * pred_score)
    union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
    return 1. - (2 * inter / union)
     

def get_geo_loss(gt_geo, pred_geo):
    d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
    d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
    area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
    area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
    w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
    h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    iou_loss_map = -torch.log((area_intersect + 1.0)/(area_union + 1.0))
    angle_loss_map = 1 - torch.cos(angle_pred - angle_gt)
    return iou_loss_map, angle_loss_map


class CharNetLoss(nn.Module):
    def __init__(self, weight_angle=10):
        super(CharNetLoss, self).__init__()
        self.weight_angle = weight_angle
        # score loss
        self.cel2 = nn.CrossEntropyLoss(weight=torch.Tensor([0.3,0.7])).cuda()  
        # self.cel2 = nn.CrossEntropyLoss().cuda()  

        # classification loss
        self.celn = nn.CrossEntropyLoss().cuda()


    def forward(self, y_pred, gt_score_w, gt_geo_w, ignored_map_w, gt_score_ch, gt_geo_ch, ignored_map_ch, gt_cls):
        '''
            Geo loss = IoU loss* weight + Angle loss


        '''
        # make sure all ignored maps are zeros
        assert ignored_map_w.sum().item() == 0 and ignored_map_ch.sum().item() == 0 
        pred_score_w, pred_geo_w, pred_word_orient, pred_score_ch, pred_geo_ch, pred_char_orient, pred_cls = y_pred
        pred_geo_w = torch.cat((pred_geo_w, pred_word_orient), dim=1)    # join B 4 H W and B 1 H W obtain B 5 H W
        pred_geo_ch = torch.cat((pred_geo_ch, pred_char_orient), dim=1)
        
        if torch.sum(gt_score_w) < 1:
            return torch.sum(pred_score_w + pred_geo_w) * 0 + torch.sum(pred_score_ch + pred_geo_ch) * 0
       
        # pred cls B 68 H W  gt_cls B 1 H W
        batch_size = pred_cls.size(0)
        num_classes = pred_cls.size(1)
        # pred_cls = pred_cls.permute(0,2,3,1)
        # gt_cls = gt_cls.permute(0,2,3,1)
        pred_cls = pred_cls.view(batch_size, num_classes, -1)  # B C H*W
        gt_cls = gt_cls.view(batch_size, -1)  # B H*W
        cls_loss = self.celn(pred_cls, gt_cls)
        # print('Character classification loss is {:.8f}'.format(cls_loss))

        iou_loss_map_w, angle_loss_map_w = get_geo_loss(gt_geo_w, pred_geo_w)

        angle_loss_w = torch.sum(angle_loss_map_w * gt_score_w) / torch.sum(gt_score_w)
        iou_loss_w = torch.sum(iou_loss_map_w * gt_score_w) / torch.sum(gt_score_w)
        geo_loss_w = self.weight_angle * angle_loss_w + iou_loss_w
        # print('Word score loss is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}'.format(score_loss_w, angle_loss_w, iou_loss_w))


        iou_loss_map_ch, angle_loss_map_ch = get_geo_loss(gt_geo_ch, pred_geo_ch)

        angle_loss_ch = torch.sum(angle_loss_map_ch * gt_score_ch) / torch.sum(gt_score_ch)
        iou_loss_ch = torch.sum(iou_loss_map_ch * gt_score_ch) / torch.sum(gt_score_ch)
        geo_loss_ch = self.weight_angle * angle_loss_ch + iou_loss_ch
        # print('Character score loss ch is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}'.format(score_loss_ch, angle_loss_ch, iou_loss_ch))

        



        # score_loss_w = get_dice_loss(gt_score_w, pred_score_w*(1-ignored_map_w))
        # gt_score_w B 1 H W pred_score_w B 2 H W
        pred_score_w = pred_score_w * (1 - ignored_map_w)
        pred_score_w = pred_score_w.view(batch_size, 2, -1)  # 2 is number of classes which are text non text
        gt_score_w = gt_score_w.view(batch_size, -1)
        score_loss_w = self.cel2(pred_score_w, gt_score_w)


        # score_loss_ch = get_dice_loss(gt_score_ch, pred_score_ch*(1-ignored_map_ch))
        pred_score_ch = pred_score_ch * (1 - ignored_map_ch)
        pred_score_ch = pred_score_ch.view(batch_size,2,-1)
        gt_score_ch = gt_score_ch.view(batch_size,-1)
        score_loss_ch = self.cel2(pred_score_ch, gt_score_ch)

        total_loss = geo_loss_w + score_loss_w + geo_loss_ch + score_loss_ch + cls_loss
        return angle_loss_w, iou_loss_w, geo_loss_w, score_loss_w, angle_loss_ch, iou_loss_ch, geo_loss_ch, score_loss_ch, total_loss, cls_loss
