import torch
from torch import nn
import numpy as np


class CharNetLoss(nn.Module):
    def __init__(self, Lambda, ratio=3, reduction='mean', weight_angle=10):
        """Implement CharNet Loss.
        """
        super(CharNetLoss, self).__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.Lambda = Lambda
        self.ratio = ratio
        self.reduction = reduction
        self.weight_angle = weight_angle

    def forward(self, outputs, labels, training_masks):
        texts = outputs[:, -1, :, :]
        kernels = outputs[:, :-1, :, :]
        gt_texts = labels[:, -1, :, :]
        gt_kernels = labels[:, :-1, :, :]

        selected_masks = self.ohem_batch(texts, gt_texts, training_masks)
        selected_masks = selected_masks.to(outputs.device)

        loss_text = self.dice_loss(texts, gt_texts, selected_masks)

        loss_kernels = []
        mask0 = torch.sigmoid(texts).data.cpu().numpy()
        mask1 = training_masks.data.cpu().numpy()
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
        selected_masks = torch.from_numpy(selected_masks).float()
        selected_masks = selected_masks.to(outputs.device)
        kernels_num = gt_kernels.size()[1]
        for i in range(kernels_num):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.dice_loss(kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.stack(loss_kernels).mean(0)
        if self.reduction == 'mean':
            loss_text = loss_text.mean()
            loss_kernels = loss_kernels.mean()
        elif self.reduction == 'sum':
            loss_text = loss_text.sum()
            loss_kernels = loss_kernels.sum()

        loss = self.Lambda * loss_text + (1 - self.Lambda) * loss_kernels

		if torch.sum(gt_score) < 1:
			return torch.sum(pred_score + pred_geo) * 0
		
		classify_loss = get_dice_loss_east(gt_score, pred_score*(1-ignored_map))
		iou_loss_map, angle_loss_map = get_geo_loss_east(gt_geo, pred_geo)

		angle_loss = torch.sum(angle_loss_map*gt_score) / torch.sum(gt_score)
		iou_loss = torch.sum(iou_loss_map*gt_score) / torch.sum(gt_score)
		geo_loss = self.weight_angle * angle_loss + iou_loss
		print('classify loss is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}'.format(classify_loss, angle_loss, iou_loss))
		return geo_loss + classify_loss


        return loss_text, loss_kernels, loss

    def dice_loss(self, input, target, mask):
        input = torch.sigmoid(input)

        input = input.contiguous().view(input.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)

        input = input * mask
        target = target * mask

        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + 0.001
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)
        return 1 - d

    def ohem_single(self, score, gt_text, training_mask):
        pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))

        if pos_num == 0:
            # selected_mask = gt_text.copy() * 0 # may be not good
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_num = (int)(np.sum(gt_text <= 0.5))
        neg_num = (int)(min(pos_num * 3, neg_num))

        if neg_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_score = score[gt_text <= 0.5]
        # 将负样本得分从高到低排序
        neg_score_sorted = np.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]
        # 选出 得分高的 负样本 和正样本 的 mask
        selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    def ohem_batch(self, scores, gt_texts, training_masks):
        scores = scores.data.cpu().numpy()
        gt_texts = gt_texts.data.cpu().numpy()
        training_masks = training_masks.data.cpu().numpy()

        selected_masks = []
        for i in range(scores.shape[0]):
            selected_masks.append(self.ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

        selected_masks = np.concatenate(selected_masks, 0)
        selected_masks = torch.from_numpy(selected_masks).float()

        return selected_masks

    def get_dice_loss_east(gt_score, pred_score):
        inter = torch.sum(gt_score * pred_score)
        union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
        return 1. - (2 * inter / union)
        

    def get_geo_loss_east(gt_geo, pred_geo):
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


class Loss(nn.Module):
	def __init__(self, ):
		super(Loss, self).__init__()
		self.weight_angle = weight_angle

	def forward(self, gt_score, pred_score, gt_geo, pred_geo, ignored_map):
