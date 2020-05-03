import numpy as np
import os
import time
import warnings
import pickle
# from accimage import Image
from PIL import Image
import io
import cv2

try:
    from fasterzip import ZipFile
    fastzip = True
except:
    warnings.warn('For faster loading of zip, you can install fasterzip via '
                  '`pip install https://github.com/TkTech/fasterzip/archive/master.zip`')
    from zipfile import ZipFile
    fastzip = False

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import scipy.io

def preprocess_words(word_ar):
    words = []
    for ii in range(np.shape(word_ar)[0]):
        s = word_ar[ii]
        start = 0
        while s[start] == ' ' or s[start] == '\n':
            start += 1
        for i in range(start + 1, len(s) + 1):
            if i == len(s) or s[i] == '\n' or s[i] == ' ':
                if start != i:
                    words.append(s[start : i])
                start = i + 1
    return words

def get_path(path):
    if fastzip:
        return path.encode()
    else:
        return path

class SynthTextDataset(Dataset):
    def __init__(self, zip_path, cache_path=None):
        self.zip_path = zip_path
        self.cache_path = cache_path

    def lazy_init(self):
        """
        we lazily initialize rather than opening the Zip file before-hand because,
        ZipFile is not thread/fork safe. If we dont lazily initialize,
        then a bunch of file read errors show up, that are false-positives (the zip itself is fine)
        """
        if hasattr(self, 'images'):
            return
        zip_path = self.zip_path
        cache_path = self.cache_path
        tm = time.time()
        self.zip = ZipFile(get_path(zip_path))
        print('opening zip file....done in {} seconds'.format(time.time() - tm))
        if cache_path is None:
            cache_path = os.path.join(os.path.dirname(zip_path), 'gt.pkl')
        tm = time.time()
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                gt = pickle.load(f)
        else:
            if fastzip:
                with self.zip.read(get_path(os.path.join('SynthText', 'gt.mat'))) as file_contents:
                    gt_ = scipy.io.loadmat(io.BytesIO(file_contents))
            else:
                file_contents = self.zip.read(get_path(os.path.join('SynthText', 'gt.mat')))
                gt_ = scipy.io.loadmat(io.BytesIO(file_contents))
            gt = {
                'imnames': gt_['imnames'],
                'wordBB' : gt_['wordBB'],
                'charBB' : gt_['charBB'],
                'txt' : gt_['txt'],
            }
            del gt_
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(gt, f, protocol=pickle.HIGHEST_PROTOCOL)
            except IOError:
                warnings.warn("Couldn't write SynthTextDataset cache at {}".format(cache_path))
        self.images = gt['imnames'][0]
        self.w_bboxes = gt['wordBB'][0]
        self.ch_bboxes = gt['charBB'][0]
        self.text = gt['txt'][0]
        del gt
        print('loading metadata done in {} seconds'.format(time.time() - tm))

    def __len__(self):
        return 858750 # size of SynthText dataset

    def __getitem__(self, index):
        self.lazy_init()
        path = str(self.images[index][0])
        w_boxes = self.w_bboxes[index]
        ch_boxes = self.ch_bboxes[index]
        words = preprocess_words(self.text[index])
        im = 'SynthText/' + path
        if len(np.shape(w_boxes)) == 2:
            w_boxes = np.array([w_boxes])
            w_boxes = np.transpose(w_boxes, (1, 2, 0))

        w_boxes = np.transpose(w_boxes, (2, 1, 0)) # num_boxes, 4 points, 2 xy

        if len(np.shape(ch_boxes)) == 2:
            ch_boxes = np.array([ch_boxes])
            ch_boxes = np.transpose(ch_boxes, (1, 2, 0))

        ch_boxes = np.transpose(ch_boxes, (2, 1, 0)) # num_boxes, 4 points, 2 xy

        try:
            if fastzip:
                with self.zip.read(get_path(im)) as imbytes:
                    # pil_img = Image(bytes(imbytes))
                    pil_img = Image.open(io.BytesIO(bytes(imbytes)))
            else:
                imbytes = self.zip.read(get_path(im))
                pil_img = Image.open(io.BytesIO(imbytes))
                # pil_img = Image(bytes(imbytes))
        except Exception as e:
            print(e, index, path)
            raise e

        return pil_img, w_boxes, ch_boxes, words



def vis(img, word_bbs, char_bbs, txts):
    img_word_ins = img.copy()
    img_word_ins2 = img.copy()
    for txt,word_bbox in zip(txts, word_bbs):
        word_bbox = word_bbox.astype(np.int32)
        cv2.polylines(img_word_ins, [word_bbox],
                      True, (0, 255, 0), 2)
        cv2.putText(
            img_word_ins,
            '{}'.format(txt),
            (word_bbox[0][0], word_bbox[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )

    chars = "".join(txts)
    for ch,ch_bbox in zip(chars, char_bbs):
        ch_bbox = ch_bbox.astype(np.int32)
        cv2.polylines(img_word_ins2, [ch_bbox],
                        True, (0, 255, 0), 2)
        cv2.putText(
            img_word_ins2,
            '{}'.format(ch),
            (ch_bbox[0][0], ch_bbox[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
    
    return img_word_ins, img_word_ins2

if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--synthtext', default='/media/end_z820_1/Yeni Birim/DATASETS/SynthText.zip', type=str,
                            help='path for synthtext dataset')
        args = parser.parse_args()
        return args

    args = parse_args()
    time1 = time.time()
    dataset = SynthTextDataset(args.synthtext)
    print('| Time taken for data init %.2f' % (time.time() - time1))
    im, wboxes, cboxes, txts = dataset[0]
    im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)  # PIL to cv2
    img_words, img_chars = vis(im, wboxes, cboxes, txts)
    cv2.imwrite("res_1.jpg", img_words)
    cv2.imwrite("res_2.jpg", img_chars)
