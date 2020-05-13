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


class CharnetLoss(nn.Module):
	def __init__(self, weight_angle=10):
		super(CharnetLoss, self).__init__()
		self.weight_angle = weight_angle

	def forward(self, gt_score_w, pred_score_w, gt_geo_w, pred_geo_w, ignored_map_w, gt_score_ch, pred_score_ch, gt_geo_ch, pred_geo_ch, ignored_map_ch):
		if torch.sum(gt_score_w) < 1:
			return torch.sum(pred_score_w + pred_geo_w) * 0 + torch.sum(pred_score_ch + pred_geo_ch) * 0
		
		classify_loss_w = get_dice_loss(gt_score_w, pred_score_w*(1-ignored_map_w))
		iou_loss_map_w, angle_loss_map_w = get_geo_loss(gt_geo_w, pred_geo_w)

		angle_loss_w = torch.sum(angle_loss_map_w*gt_score_w) / torch.sum(gt_score_w)
		iou_loss_w = torch.sum(iou_loss_map_w*gt_score_w) / torch.sum(gt_score_w)
		geo_loss_w = self.weight_angle * angle_loss_w + iou_loss_w
		print('Word classify loss is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}'.format(classify_loss_w, angle_loss_w, iou_loss_w))

		classify_loss_ch = get_dice_loss(gt_score_ch, pred_score_ch*(1-ignored_map_ch))
		iou_loss_map_ch, angle_loss_map_ch = get_geo_loss(gt_geo_ch, pred_geo_ch)

		angle_loss_ch = torch.sum(angle_loss_map_ch*gt_score_ch) / torch.sum(gt_score_ch)
		iou_loss_ch = torch.sum(iou_loss_map_ch*gt_score_ch) / torch.sum(gt_score_ch)
		geo_loss_ch = self.weight_angle * angle_loss_ch + iou_loss_ch
        print('Character classify loss ch is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}'.format(classify_loss_ch, angle_loss_ch, iou_loss_ch))
		total_loss = geo_loss_w + classify_loss_w + geo_loss_ch + classify_loss_ch
		return angle_loss_w, iou_loss_w, geo_loss_w, classify_loss_w, angle_loss_ch, iou_loss_ch, geo_loss_ch, classify_loss_ch, total_loss
