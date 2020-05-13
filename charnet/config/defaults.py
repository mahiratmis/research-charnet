# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from yacs.config import CfgNode as CN


_C = CN()

_C.INPUT_SIZE = 2280
_C.SIZE_DIVISIBILITY = 1
_C.WEIGHT= ""

_C.CHAR_DICT_FILE = "datasets/ICDAR2015/test/char_dict.txt"
_C.WORD_LEXICON_PATH =  "datasets/ICDAR2015/test/GenericVocabulary.txt"

_C.WORD_MIN_SCORE = 0.95
_C.WORD_NMS_IOU_THRESH = 0.15
_C.CHAR_MIN_SCORE = 0.25
_C.CHAR_NMS_IOU_THRESH = 0.3
_C.MAGNITUDE_THRESH = 0.2

_C.WORD_STRIDE = 4
_C.CHAR_STRIDE = 4
_C.NUM_CHAR_CLASSES = 68

_C.WORD_DETECTOR_DILATION = 1
_C.RESULTS_SEPARATOR = chr(31)

_C.trainroot_icdar = '/media/end_z820_1/Yeni Birim/DATASETS/ICDAR_2015/incidental/e2e/train'
_C.testroot_icdar = '/media/end_z820_1/Yeni Birim/DATASETS/ICDAR_2015/incidental/e2e/test'
_C.output_dir_icdar = '/media/end_z820_1/Yeni Birim/DATASETS/ICDAR_2015/incidental/e2e/output'
_C.data_shape = 640
_C.trainroot_synth= '/media/end_z820_1/Yeni Birim/DATASETS/SynthText/SynthText.zip'
_C.testroot_synth= '/media/end_z820_1/Yeni Birim/DATASETS/ICDAR_2015/incidental/e2e/test'
_C.output_dir_synth= '/media/end_z820_1/Yeni Birim/DATASETS/ICDAR_2015/incidental/e2e/output'

_C.validation_split = 0.2

# train config
_C.gpu_id = '0'
_C.workers = 8
_C.start_epoch = 0
_C.epochs = 600
_C.train_batch_size = 4
_C.lr = 1e-4
_C.end_lr = 1e-7
_C.lr_gamma = 0.1
_C.lr_decay_step = [200,400]
_C.weight_decay = 5e-4
_C.warm_up_epoch = 6
_C.warm_up_lr = _C.lr * _C.lr_gamma

_C.display_input_images = False
_C.display_output_images = False
_C.display_interval = 10
_C.show_images_interval = 50

_C.pretrained = True
_C.restart_training = True
_C.checkpoint = ''

# net config
_C.n = 6
_C.m = 0.5
_C.seed = 60
_C.OHEM_ratio = 3
_C.scale = 1
