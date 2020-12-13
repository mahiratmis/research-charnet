import cv2
import os
from charnet.config import cfg as config

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

import shutil
import glob
import time
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import torch.utils.data as Data
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler

from datasets.synth_dataset import SynthTextDataset, my_collate
from charnet.modeling.model import CharNet
from charnet.modeling.loss import CharNetLoss
from charnet.modeling.utils import load_checkpoint, save_checkpoint, setup_logger

from utils.cal_recall import cal_recall_precison_f1


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# learning rate的warming up操作
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < config.warm_up_epoch:
        lr = 1e-6 + (config.lr - 1e-6) * epoch / (config.warm_up_epoch)
    else:
        lr = config.lr * (config.lr_gamma ** (epoch / config.lr_decay_step[0]))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train_epoch(net, optimizer, scheduler, train_loader, device, criterion, epoch, all_step, writer, logger):
    net.train()
    train_loss = 0.
    start = time.time()
    scheduler.step()
    accumulate_iter = 1
    min_loss = 1e3
    pth_idx = 0
    # lr = adjust_learning_rate(optimizer, epoch)
    lr = scheduler.get_last_lr()[0]
    for i, (images, score_w, geo_w, ignored_w, score_ch, geo_ch, ignored_ch, w_boxes, ch_boxes, words, class_map, paths) in enumerate(train_loader):
        
        cur_batch = images.size()[0]
        images = images.to(device)
        score_w, geo_w, ignored_w = score_w.to(device), geo_w.to(device), ignored_w.to(device)
        score_ch, geo_ch, ignored_ch = score_ch.to(device), geo_ch.to(device), ignored_ch.to(device)
        class_map = class_map.to(device)
        # Forward
        y1 = net(images, 1, 1, 512, 512)
        angle_loss_w, iou_loss_w, geo_loss_w, score_loss_w, angle_loss_ch, iou_loss_ch, geo_loss_ch, score_loss_ch, loss, cls_loss = criterion(y1, score_w, geo_w, ignored_w, score_ch, geo_ch, ignored_ch, class_map)
        
        # pred_score_w, pred_geo_w, pred_angle_w, pred_score_ch, pred_geo_ch, pred_angle_ch, _ = y1
        # for ix in range(4):
        #     print("Pred geo min{} max{} -- Gt min{} max{}".format(pred_geo_ch[ix].min(), pred_geo_ch[ix].max(), geo_ch[ix].min(), geo_ch[ix].max()))
        # print("Pred angle min{} max{} -- Gt angle min{} max{}\n".format(pred_angle_ch[4].min(), pred_geo_ch[4].max(), geo_ch[4].min(), geo_ch[4].max()))

        
        # Backward
        
        # accumulate gradients accumulate_iter iters
        loss /= accumulate_iter
        if i % accumulate_iter == accumulate_iter - 1:
            optimizer.zero_grad()
        loss.backward()
        if i % accumulate_iter == accumulate_iter - 1:
            optimizer.step()
        train_loss += loss.item()

        angle_loss_w = angle_loss_w.item()
        iou_loss_w = iou_loss_w.item()
        geo_loss_w = geo_loss_w.item()
        score_loss_w = score_loss_w.item()
        
        angle_loss_ch = angle_loss_ch.item()
        iou_loss_ch = iou_loss_ch.item()
        geo_loss_ch = geo_loss_ch.item()
        score_loss_ch = score_loss_ch.item()
        cls_loss = cls_loss.item()

        loss = loss.item()
        if loss < min_loss:
            min_loss = loss
            save_checkpoint(f"{config.output_dir_synth}/{pth_idx%4}.pth", net, optimizer, epoch, min_loss, logger)
            pth_idx += 1

        # pred_score_w, pred_geo_w, pred_angle_w, pred_score_ch, pred_geo_ch, pred_angle_ch, _ = y1
        # print("char gt pred ", geo_ch.min(), pred_geo_ch.min(), geo_ch.max(), pred_geo_ch.max())
        # print("word gt pred ", geo_w.min(), pred_geo_w.min(), geo_w.max(), pred_geo_w.max())
        # print("word gt pred ", geo_w.min(), pred_geo_w.min(), geo_w.max(), pred_geo_w.max())
        # print("word gt pred angle", geo_w.min(), pred_geo_w.min(), geo_w.max(), pred_geo_w.max())
        cur_step = epoch * all_step + i

        writer.add_scalar(tag='Train/angle_loss_w', scalar_value=angle_loss_w, global_step=cur_step)
        writer.add_scalar(tag='Train/iou_loss_w', scalar_value=iou_loss_w, global_step=cur_step)
        writer.add_scalar(tag='Train/geo_loss_w', scalar_value=geo_loss_w, global_step=cur_step)
        writer.add_scalar(tag='Train/score_loss_w', scalar_value=score_loss_w, global_step=cur_step)
        writer.add_scalar(tag='Train/angle_loss_ch', scalar_value=angle_loss_ch, global_step=cur_step)
        writer.add_scalar(tag='Train/iou_loss_ch', scalar_value=iou_loss_ch, global_step=cur_step)
        writer.add_scalar(tag='Train/geo_loss_ch', scalar_value=geo_loss_ch, global_step=cur_step)
        writer.add_scalar(tag='Train/score_loss_ch', scalar_value=score_loss_ch, global_step=cur_step)
        writer.add_scalar(tag='Train/classification_loss', scalar_value=cls_loss, global_step=cur_step)        
        writer.add_scalar(tag='Train/total_loss', scalar_value=loss, global_step=cur_step)
        writer.add_scalar(tag='Train/lr', scalar_value=lr, global_step=cur_step)

        batch_loss_score = score_loss_w + score_loss_ch 
        batch_loss_geo = geo_loss_w + geo_loss_ch
        batch_loss_cls = cls_loss

        if i % config.display_interval == 0:
            batch_time = time.time() - start
            logger.info(
                '[{}/{}], [{}/{}], step: {}, {:.3f} samples/sec, batch_loss: {:.4f}, batch_loss_score: {:.4f}, batch_loss_geo: {:.4f}, batch_loss_cls:{:.4f}, time:{:.4f}, lr:{}'.format(
                    epoch, config.epochs, i, all_step, cur_step, config.display_interval * cur_batch / batch_time,
                    loss, batch_loss_score, batch_loss_geo, batch_loss_cls, batch_time, lr))
            start = time.time()

        if i % config.show_images_interval == 0:
            if config.display_input_images:
                # show images on tensorboard
                x = vutils.make_grid(images.detach().cpu(), nrow=4, normalize=True, scale_each=True, padding=20)
                writer.add_image(tag='input/image', img_tensor=x, global_step=cur_step)


            if config.display_output_images:
                print("logging images")
                pred_score_w, pred_geo_w, pred_angle_w, pred_score_ch, pred_geo_ch, pred_angle_ch, _ = y1

                y1 = pred_score_w
                show_y = y1.detach().cpu()
                b, c, h, w = show_y.size()
                show_y = show_y.reshape(b * c, h, w)
                show_y = vutils.make_grid(show_y.unsqueeze(1), nrow=config.n, normalize=True, padding=20, pad_value=1)
                writer.add_image(tag='output/0preds_w_score', img_tensor=show_y, global_step=cur_step)

                y1 = score_w
                show_y = y1.detach().cpu()
                b, c, h, w = show_y.size()
                show_y = show_y.reshape(b * c, h, w)
                show_y = vutils.make_grid(show_y.unsqueeze(1), nrow=config.n, normalize=False, padding=20, pad_value=1)
                writer.add_image(tag='output/0gt_w_score', img_tensor=show_y, global_step=cur_step)
                

                y1 = pred_score_ch
                show_y = y1.detach().cpu()
                b, c, h, w = show_y.size()
                show_y = show_y.reshape(b * c, h, w)
                show_y = vutils.make_grid(show_y.unsqueeze(1), nrow=config.n, normalize=True, padding=20, pad_value=1)
                writer.add_image(tag='output/1pred_ch_score', img_tensor=show_y, global_step=cur_step)

                y1 = score_ch
                show_y = y1.detach().cpu()
                b, c, h, w = show_y.size()
                show_y = show_y.reshape(b * c, h, w)
                show_y = vutils.make_grid(show_y.unsqueeze(1), nrow=config.n, normalize=False, padding=20, pad_value=1)
                writer.add_image(tag='output/1gt_ch_score', img_tensor=show_y, global_step=cur_step)

                y1 = pred_geo_ch
                show_y = y1.detach().cpu()
                b, c, h, w = show_y.size()
                show_y = show_y.reshape(b * c, h, w)
                show_y = vutils.make_grid(show_y.unsqueeze(1), nrow=config.n, normalize=True, padding=20, pad_value=1)
                writer.add_image(tag='output/2pred_ch_geo', img_tensor=show_y, global_step=cur_step)

                y1 = geo_ch
                show_y = y1.detach().cpu()
                b, c, h, w = show_y.size()
                show_y = show_y.reshape(b * c, h, w)
                show_y = vutils.make_grid(show_y.unsqueeze(1), nrow=config.n, normalize=True, padding=20, pad_value=1)
                writer.add_image(tag='output/2gt_ch_geo', img_tensor=show_y, global_step=cur_step)

                y1 = pred_geo_w
                show_y = y1.detach().cpu()
                b, c, h, w = show_y.size()
                show_y = show_y.reshape(b * c, h, w)
                show_y = vutils.make_grid(show_y.unsqueeze(1), nrow=config.n, normalize=True, padding=20, pad_value=1)
                writer.add_image(tag='output/3pred_w_geo', img_tensor=show_y, global_step=cur_step)

                y1 = geo_w
                show_y = y1.detach().cpu()
                b, c, h, w = show_y.size()
                show_y = show_y.reshape(b * c, h, w)
                show_y = vutils.make_grid(show_y.unsqueeze(1), nrow=config.n, normalize=True, padding=20, pad_value=1)
                writer.add_image(tag='output/3gt_w_geo', img_tensor=show_y, global_step=cur_step)

                y1 = pred_angle_w
                show_y = y1.detach().cpu()
                b, c, h, w = show_y.size()
                show_y = show_y.reshape(b * c, h, w)
                show_y = vutils.make_grid(show_y.unsqueeze(1), nrow=config.n, normalize=True, padding=20, pad_value=1)
                writer.add_image(tag='output/4pred_w_angle', img_tensor=show_y, global_step=cur_step)

                y1 = pred_angle_ch
                show_y = y1.detach().cpu()
                b, c, h, w = show_y.size()
                show_y = show_y.reshape(b * c, h, w)
                show_y = vutils.make_grid(show_y.unsqueeze(1), nrow=config.n, normalize=True, padding=20, pad_value=1)
                writer.add_image(tag='output/4pred_ch_angle', img_tensor=show_y, global_step=cur_step)

    writer.add_scalar(tag='Train_epoch/loss', scalar_value=train_loss / all_step, global_step=epoch)
    return train_loss / all_step, lr


def eval(model, save_path, val_loader, device, test_path="/media/end_z820_1/Yeni Birim/DATASETS/a/SynthText/SynthText"):
    model.eval()
    img_path = os.path.join(test_path, 'img')
    gt_path = os.path.join(test_path, 'gt')
    save_path = os.path.join(test_path, 'pred')
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for im, score_w, geo_w, ignored_w, score_ch, geo_ch, ignored_ch, w_boxes, ch_boxes, words, class_map, paths in val_loader:
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        for path, img, wboxes, cboxes,wrds, cls_map in zip(paths, im,w_boxes,ch_boxes,words, class_map):
            img_path = os.path.join(test_path, path) 
            img_name = os.path.basename(img_path).split('.')[0]
            with torch.no_grad():
                im = im.to(device)
                y1 = model(im, 1, 1, 512, 512)
                char_bboxes, char_scores, word_instances = model.get_predictions(*y1, 1, 1, 512, 512)
                if len(word_instances) > 0:
                    save_word_recognition(word_instances, img_name, save_path)
    result_dict = cal_recall_precison_f1(gt_path, save_path)
    return result_dict['recall'], result_dict['precision'], result_dict['hmean']

def save_word_recognition(word_instances, image_id, save_root, separator=','):
    with open('{}/res_{}.txt'.format(save_root, image_id), 'wt') as fw:
        for word_ins in word_instances:
            if len(word_ins.text) > 0:
                fw.write(separator.join([str(_) for _ in word_ins.word_bbox.flat]))
                fw.write('\n')


def eval_org(model, save_path, test_path, device):
    model.eval()
    # torch.cuda.empty_cache()  # speed up evaluating after training finished
    img_path = os.path.join(test_path, 'img')
    gt_path = os.path.join(test_path, 'gt')
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    long_size = 2240
    # 预测所有测试图片
    img_paths = [os.path.join(img_path, x) for x in os.listdir(img_path)]
    for img_path in tqdm(img_paths, desc='test models'):
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_path, 'res_' + img_name + '.txt')

        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        #if max(h, w) > long_size:
        scale = long_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(device)
        with torch.no_grad():
            preds = model(tensor)
            preds, boxes_list = pse_decode(preds[0], config.scale)
            scale = (preds.shape[1] * 1.0 / w, preds.shape[0] * 1.0 / h)
            if len(boxes_list):
                boxes_list = boxes_list / scale
        np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')
    # 开始计算 recall precision f1
    result_dict = cal_recall_precison_f1(gt_path, save_path)
    return result_dict['recall'], result_dict['precision'], result_dict['hmean']


def main():
    if config.output_dir is None:
        config.output_dir = 'output'
    if config.restart_training:
        shutil.rmtree(config.output_dir, ignore_errors=True)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    logger = setup_logger(os.path.join(config.output_dir, 'train_log'))
    logger.info(config)
    # reproducability
    torch.manual_seed(config.seed) 
    np.random.seed(config.seed) 
    if config.gpu_id is not None and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info('train with gpu {} and pytorch {}'.format(config.gpu_id, torch.__version__))
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(config.seed) 
        torch.cuda.manual_seed_all(config.seed) 
    else:
        logger.info('train with cpu and pytorch {}'.format(torch.__version__))
        device = torch.device("cpu")


    dataset = SynthTextDataset(config.trainroot_synth)

    # Creating data indices for training and validation splits:
    test_size = int(config.validation_split * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.train_batch_size, shuffle=True, collate_fn=my_collate)
    validation_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.train_batch_size,shuffle=True, collate_fn=my_collate)


    config.output_dir = config.output_dir_synth

    writer = SummaryWriter(config.output_dir)
    model = CharNet()
    if not config.pretrained and not config.restart_training:
        model.apply(weights_init)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    # dummy_input = torch.autograd.Variable(torch.Tensor(1, 3, 600, 800).to(device))
    # writer.add_graph(models=models, input_to_model=dummy_input)
    criterion = CharNetLoss()
    # optimizer = torch.optim.SGD(models.parameters(), lr=config.lr, momentum=0.99)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    if config.checkpoint != '' and not config.restart_training:
        # start_epoch = load_checkpoint(config.checkpoint, model, logger, device, optimizer)
        checkpoint = torch.load(config.checkpoint, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        start_epoch = 0
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma)
        scheduler.last_epoch = start_epoch
    else:
        start_epoch = config.start_epoch
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma)

    all_step = len(train_loader)
    logger.info('train dataset has {} samples,{} in dataloader'.format(train_size, all_step))
    epoch = 0
    best_model = {'recall': 0, 'precision': 0, 'f1': 0, 'models': ''}
    train_loss = 0
    try:
        for epoch in range(start_epoch, config.epochs):
            start = time.time()
            train_loss, lr = train_epoch(model, optimizer, scheduler, train_loader, device, criterion, epoch, all_step,
                                         writer, logger)
            logger.info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
                epoch, config.epochs, train_loss, time.time() - start, lr))
            net_save_path = '{}/CharNet{}_loss{:.6f}_bs{}_lr{}_wd{}.pth'.format(config.output_dir, epoch,
                                                                                          train_loss, config.train_batch_size, lr,config.lr_decay_step)
            save_checkpoint(net_save_path, model, optimizer, epoch, train_loss, logger)
            if (0.3 < train_loss < 0.4 and epoch % 4 == 0) or train_loss < 0.3 or epoch==0:
                recall, precision, f1 = eval(model, os.path.join(config.output_dir, 'output'), validation_loader, device)                
                # recall, precision, f1 = (1,2,3)
                logger.info('test: recall: {:.6f}, precision: {:.6f}, f1: {:.6f}'.format(recall, precision, f1))

                net_save_path = '{}/CharNet_{}_loss{:.6f}_r{:.6f}_p{:.6f}_f1{:.6f}.pth'.format(config.output_dir, epoch,
                                                                                              train_loss,
                                                                                              recall,
                                                                                              precision,
                                                                                              f1)
                save_checkpoint(net_save_path, model, optimizer, epoch, train_loss, logger)
                if f1 > best_model['f1']:
                    best_path = glob.glob(config.output_dir + '/Best_*.pth')
                    for b_path in best_path:
                        if os.path.exists(b_path):
                            os.remove(b_path)

                    best_model['recall'] = recall
                    best_model['precision'] = precision
                    best_model['f1'] = f1
                    best_model['models'] = net_save_path

                    best_save_path = '{}/Best_{}_r{:.6f}_p{:.6f}_f1{:.6f}.pth'.format(config.output_dir, epoch,
                                                                                      recall,
                                                                                      precision,
                                                                                      f1)
                    if os.path.exists(net_save_path):
                        shutil.copyfile(net_save_path, best_save_path)
                    else:
                        save_checkpoint(best_save_path, model, optimizer, epoch, train_loss, logger)

                    pse_path = glob.glob(config.output_dir + '/CharNet_*.pth')
                    for p_path in pse_path:
                        if os.path.exists(p_path):
                            os.remove(p_path)

                writer.add_scalar(tag='Test/recall', scalar_value=recall, global_step=epoch)
                writer.add_scalar(tag='Test/precision', scalar_value=precision, global_step=epoch)
                writer.add_scalar(tag='Test/f1', scalar_value=f1, global_step=epoch)
        writer.close()
    except KeyboardInterrupt:
        save_checkpoint('{}/final.pth'.format(config.output_dir), model, optimizer, epoch, train_loss, logger)
    finally:
        if best_model['models']:
            logger.info(best_model)


if __name__ == '__main__':
    main()
