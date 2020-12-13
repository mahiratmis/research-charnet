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
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
from torchvision.utils import save_image

import scipy.io

from shapely.geometry import Polygon
import math
import matplotlib.pyplot as plt


from charnet.modeling.postprocessing import load_char_dicts
from charnet.config import cfg

import matplotlib.cm as cm 
rgb_colors = (cm.rainbow(np.linspace(0, 1, 68))[:,:3]*255).astype(int)



def get_featuremap_scales(h, w, ratio=0.25):
    '''
    Input: 
        h     : original image height
        w     : original image width
        ratio : featuremap / resized image (not original image)
    Output:        
        (scale_h, scale_w,image_resize_height,image_resize_width)
    '''
    scale = max(h, w) / float(cfg.INPUT_SIZE)
    image_resize_height = int(round(h / scale / cfg.SIZE_DIVISIBILITY) * cfg.SIZE_DIVISIBILITY)
    image_resize_width = int(round(w / scale / cfg.SIZE_DIVISIBILITY) * cfg.SIZE_DIVISIBILITY)
    scale_h = (image_resize_height * ratio ) / h
    scale_w = (image_resize_width * ratio )  / w 
    return (scale_h, scale_w,image_resize_height,image_resize_width)


def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def move_points(vertices, index1, index2, r, coef):
    '''move the two points to shrink edge
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        index1  : offset of point1
        index2  : offset of point2
        r       : [r1, r2, r3, r4] in paper
        coef    : shrink ratio in paper
    Output:
        vertices: vertices where one edge has been shinked
    '''
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1
    
    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:	
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x) 
        vertices[y1_index] += ratio * (-length_y) 
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x 
        vertices[y2_index] += ratio * length_y
    return vertices	


def shrink_poly(vertices, coef=0.3):
    '''shrink the text region
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        coef    : shrink ratio in paper
    Output:
        v       : vertices of shrinked text region <numpy.ndarray, (8,)>
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1,y1,x2,y2), cal_distance(x1,y1,x4,y4))
    r2 = min(cal_distance(x2,y2,x1,y1), cal_distance(x2,y2,x3,y3))
    r3 = min(cal_distance(x3,y3,x2,y2), cal_distance(x3,y3,x4,y4))
    r4 = min(cal_distance(x4,y4,x1,y1), cal_distance(x4,y4,x3,y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if cal_distance(x1,y1,x2,y2) + cal_distance(x3,y3,x4,y4) > \
       cal_distance(x2,y2,x3,y3) + cal_distance(x1,y1,x4,y4):
        offset = 0 # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1 # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v


def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:	
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4,2)).T
    if anchor is None:
        anchor = v[:,:1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max


def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err	


def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle.
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list: 
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)
    
    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k : area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


def is_cross_text(start_loc, length, vertices):
    '''check if the crop image crosses text regions
    Input:
        start_loc: left-top position
        length   : length of crop image
        vertices : vertices of text regions <numpy.ndarray, (n,8)>
    Output:
        True if crop image crosses text region
    '''
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    a = np.array([start_w, start_h, start_w + length, start_h, \
          start_w + length, start_h + length, start_w, start_h + length]).reshape((4,2))
    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4,2))).convex_hull
        inter = p1.intersection(p2).area
        if 0.01 <= inter / p2.area <= 0.99: 
            return True
    return False
        

def crop_img(img, vertices, labels, length, char_vertices=None):
    '''crop img patches to obtain batch and augment
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        length      : length of cropped image region
    Output:
        region            : cropped image region
        new_vertices      : new vertices in cropped region
        new_char_vertices : new char vertices in cropped region
    '''
    h, w = img.height, img.width
    # confirm the shortest side of image >= length
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)
    ratio_w = img.width / w
    ratio_h = img.height / h
    assert(ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:,[0,2,4,6]] = vertices[:,[0,2,4,6]] * ratio_w
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * ratio_h

    new_char_vertices = None
    if char_vertices is not None:
        new_char_vertices = np.zeros(char_vertices.shape)
        if char_vertices.size > 0:
            new_char_vertices[:,[0,2,4,6]] = char_vertices[:,[0,2,4,6]] * ratio_w
            new_char_vertices[:,[1,3,5,7]] = char_vertices[:,[1,3,5,7]] * ratio_h

    # find random position
    remain_h = img.height - length
    remain_w = img.width - length
    flag = True
    cnt = 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text([start_w, start_h], length, new_vertices[labels==1,:])
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices, new_char_vertices	
    
    new_vertices[:,[0,2,4,6]] -= start_w
    new_vertices[:,[1,3,5,7]] -= start_h
    if new_char_vertices is not None:
        new_char_vertices[:,[0,2,4,6]] -= start_w
        new_char_vertices[:,[1,3,5,7]] -= start_h
    return region, new_vertices, new_char_vertices


def resize_img(img, vertices, width, height, char_vertices=None):
    '''resize image and corresponding bounding boxes
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        width       : new width of image 
        height      : new height of image
    Output:
        img               : resized image
        new_vertices      : new vertices in resized image
        new_char_vertices : new char vertices in resized image
    '''
    h, w = img.height, img.width
    img = img.resize((width, height), Image.BILINEAR)

    ratio_w = width / w
    ratio_h = height / h

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:,[0,2,4,6]] = vertices[:,[0,2,4,6]] * ratio_w
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * ratio_h

    new_char_vertices = None
    if char_vertices is not None:
        new_char_vertices = np.zeros(char_vertices.shape)
        if char_vertices.size > 0:
            new_char_vertices[:,[0,2,4,6]] = char_vertices[:,[0,2,4,6]] * ratio_w
            new_char_vertices[:,[1,3,5,7]] = char_vertices[:,[1,3,5,7]] * ratio_h
    
    return img, new_vertices, new_char_vertices



def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''get rotated locations of all pixels for next stages
    Input:
        rotate_mat: rotatation matrix
        anchor_x  : fixed x position
        anchor_y  : fixed y position
        length    : length of image
    Output:
        rotated_x : rotated x positions <numpy.ndarray, (length,length)>
        rotated_y : rotated y positions <numpy.ndarray, (length,length)>
    '''
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                                                   np.array([[anchor_x], [anchor_y]])
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


def adjust_height(img, vertices, ratio=0.2, char_vertices=None):
    '''adjust height of image to aug data
    Input:
        img           : PIL Image
        vertices      : vertices of text regions <numpy.ndarray, (n,8)>
        ratio         : height changes in [0.8, 1.2]
        char_vertices : vertices of char regions <numpy.ndarray, (m,8)>
    Output:
        img               : adjusted PIL Image
        new_vertices      : adjusted vertices
        new_char_vertices : adjusted char vertices
    '''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)
    
    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * (new_h / old_h)

    new_char_vertices = None
    if char_vertices is not None:
        new_char_vertices = char_vertices.copy()
        if char_vertices.size > 0:
            new_char_vertices[:,[1,3,5,7]] = char_vertices[:,[1,3,5,7]] * (new_h / old_h)
    return img, new_vertices, new_char_vertices


def rotate_img(img, vertices, angle_range=10, char_vertices=None):
    '''rotate image [-10, 10] degree to aug data
    Input:
        img           : PIL Image
        vertices      : vertices of text regions <numpy.ndarray, (n,8)>
        angle_range   : rotate range
        char_vertices : vertices of char regions <numpy.ndarray, (m,8)>
    Output:
        img               : rotated PIL Image
        new_vertices      : rotated vertices
        new_char_vertices : rotated char vertices
    '''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i,:] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x],[center_y]]))
    
    new_char_vertices = None
    if char_vertices is not None:
        new_char_vertices = np.zeros(char_vertices.shape)
        for i, vertice in enumerate(char_vertices):
            new_char_vertices[i,:] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x],[center_y]]))  

    return img, new_vertices, new_char_vertices


def get_score_geo(img, vertices, labels, scale, length, classes=None, word_lens=None):
    '''generate score gt and geometry gt
    Input:
        img     : PIL Image
        vertices: vertices of text regions <numpy.ndarray, (n,8)>
        labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        scale   : feature map / image
        length  : image length
    Output:
        score gt, geo gt, ignored
    '''
    score_map   = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
    geo_map     = np.zeros((int(img.height * scale), int(img.width * scale), 5), np.float32)
    ignored_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
    class_map   = np.zeros((1, int(img.height * scale), int(img.width * scale)), np.float32)

    #score_map_char   = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
    #geo_map_char     = np.zeros((int(img.height * scale), int(img.width * scale), 5), np.float32)
    
    index = np.arange(0, length, int(1/scale))
    index_x, index_y = np.meshgrid(index, index)
    ignored_polys = []
    polys = []
    #poly_chars = []
    
    #start=0
    for i, (vertice, cls) in enumerate(zip(vertices,classes)):
        #end = word_lens[i]
        if labels[i] == 0:
            ignored_polys.append(np.around(scale * vertice.reshape((4,2))).astype(np.int32))
            #start = end
            continue		
        
        
        poly = np.around(scale * shrink_poly(vertice).reshape((4,2))).astype(np.int32) # scaled & shrinked        
        polys.append(poly)
        temp_mask = np.zeros(score_map.shape[:-1], np.float32)
        cv2.fillPoly(temp_mask, [poly], 1)
        
        theta = find_min_rect_angle(vertice)
        rotate_mat = get_rotate_mat(theta)
        
        rotated_vertices = rotate_vertices(vertice, theta)
        x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
        rotated_x, rotated_y = rotate_all_pixels(rotate_mat, vertice[0], vertice[1], length)
    
        d1 = rotated_y - y_min
        d1[d1 < 0] = 0
        d2 = y_max - rotated_y
        d2[d2 < 0] = 0
        d3 = rotated_x - x_min
        d3[d3 < 0] = 0
        d4 = x_max - rotated_x
        d4[d4 < 0] = 0
        geo_map[:,:,0] += d1[index_y, index_x] * temp_mask
        geo_map[:,:,1] += d2[index_y, index_x] * temp_mask
        geo_map[:,:,2] += d3[index_y, index_x] * temp_mask
        geo_map[:,:,3] += d4[index_y, index_x] * temp_mask
        geo_map[:,:,4] += theta * temp_mask
        class_map += cls * temp_mask
        class_map[class_map > 67] = 0  # set intersections to zero

        #start = end
    
    cv2.fillPoly(ignored_map, ignored_polys, 1)
    cv2.fillPoly(score_map, polys, 1)
    # print("-" * 50)
    score_map = torch.LongTensor(score_map).permute(2,0,1)
    # for i in range(5):
    #     print(f"{i} min {geo_map[:,:,i].min()} max {geo_map[:,:,i].max()}")
    geo_map = torch.Tensor(geo_map).permute(2,0,1)
    ignored_map = torch.Tensor(ignored_map).permute(2,0,1)
    class_map = torch.LongTensor(class_map)
    return score_map, geo_map, ignored_map, class_map


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
                    words.append(s[start:i])
                start = i + 1
    return words


def get_path(path):
    if fastzip:
        return path.encode()
    else:
        return path


def my_collate(batch):
    pil_img       = torch.stack([item[0] for item in batch], dim=0) 
    score_w       = torch.stack([item[1] for item in batch], dim=0)
    geo_w         = torch.stack([item[2] for item in batch], dim=0)
    ignored_w     = torch.stack([item[3] for item in batch], dim=0)
    score_ch      = torch.stack([item[4] for item in batch], dim=0)
    geo_ch        = torch.stack([item[5] for item in batch], dim=0) 
    ignored_ch    = torch.stack([item[6] for item in batch], dim=0) 
    w_boxes       = [item[7] for item in batch]
    ch_boxes      = [item[8] for item in batch]
    word_indices  = [item[9] for item in batch]
    class_maps    = torch.stack([item[10] for item in batch], dim=0) 
    paths         = [item[11] for item in batch]
    return [pil_img, score_w, geo_w, ignored_w, score_ch, geo_ch, ignored_ch, w_boxes, ch_boxes , word_indices, class_maps, paths]


class SynthTextDataset(Dataset):
    def __init__(self, zip_path, cache_path=None, scale=0.25, length=512):
        self.zip_path = zip_path
        self.cache_path = cache_path
        self.idx=0
        self.ch_idx=0
        self.word_to_idx = {}
        self.char_to_idx = {}
        self.words=[]
        self.chars=[]
        self.scale= scale    # featuremap size / image size
        self.length = length  # image dimensions after crop    
        self.idx_to_char, self.char_to_idx = load_char_dicts(cfg.CHAR_DICT_FILE)    

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
        self.images = gt['imnames'][0][:5000]
        self.w_bboxes = gt['wordBB'][0][:5000]
        self.ch_bboxes = gt['charBB'][0][:5000]
        self.text = gt['txt'][0][:5000]
        del gt
        print('loading metadata done in {} seconds'.format(time.time() - tm))

    def update_word_to_idx(self, words):
        for w in set(words):
            if w not in self.word_to_idx:
                self.words.append(w)
                self.word_to_idx[w] = self.idx
                self.idx += 1
                # for ch in set(w):
                #     if ch not in self.char_to_idx:
                #         self.chars.append(ch)
                #         self.char_to_idx[ch] = self.ch_idx
                #         self.ch_idx += 1

    def words_to_indices(self, words):
        return [self.word_to_idx[w] for w in words]

    def get_words(self, indices):
        return [self.words[idx] for idx in indices]

    def words_to_char_indices(self, words):
        return [[self.char_to_idx[ch.upper()] for ch in w] for w in words]

    def words_to_char_indices_flattened(self, words):
        return [self.char_to_idx[ch.upper()] for w in words for ch in w]

    def __len__(self):
        return 5000   # 858750 size of SynthText dataset

    def __getitem__(self, index):
        '''Checks if any of the points of bbox is inside image
        Input:
            index    : index of the image to retrieve target information <int>
        Output:
            each of the Torch.Tensors is a float tensor
            pil_img     : image at index <Torch.Tensor (h,w)>
            score_w     : text non text word probabilities  <Torch.Tensor (h*scale,w*scale)>
            geo_w       : geometry information of word bboxes <Torch.Tensor (h*scale,w*scale,5)>
            ignored_w   : unreadable word regions <Torch.Tensor (h*scale,w*scale)>
            score_ch    : text non text char probabilities  <Torch.Tensor (h*scale,w*scale)>
            geo_ch      : geometry information of char bboxes <Torch.Tensor (h*scale,w*scale,5)>
            ignored_ch  : unreadable char regions <Torch.Tensor (h*scale,w*scale)>
            w_boxes     : bbox coors of words <Torch.Tensor (m,8)>
            ch_boxes    : bbox coors of chars <Torch.Tensor (n,8)> 
            word_indices: number correspondings of words <Torch.LongTensor (m,1)>
            class_map   : class labeled map <Torch.LongTensor (h*scale, w*scale)>
        '''        
        self.lazy_init()
        path = str(self.images[index][0])
        w_boxes = self.w_bboxes[index]
        ch_boxes = self.ch_bboxes[index]
        words = preprocess_words(self.text[index])
        
        word_lens =[len(w) for w in words]
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

        img_newsize = 512
        w_boxes = w_boxes.reshape(-1,8)
        ch_boxes = ch_boxes.reshape(-1,8)
        labels_readable = [1 for _ in range(w_boxes.shape[0])]
        # pil_img, w_boxes, ch_boxes = adjust_height(pil_img, w_boxes, char_vertices=ch_boxes) 
        pil_img, w_boxes, ch_boxes = rotate_img(pil_img, w_boxes, char_vertices=ch_boxes)
        pil_img, w_boxes, ch_boxes = crop_img(pil_img, w_boxes, [1 for _ in range(w_boxes.shape[0])], img_newsize, char_vertices=ch_boxes)    
        w_boxes, ch_boxes, words = detect_out_of_region_bboxes((pil_img.height, pil_img.width), w_boxes, ch_boxes, words)
        self.update_word_to_idx(words)        
        #pil_img, w_boxes, ch_boxes = resize_img(pil_img, w_boxes, img_newsize, img_newsize, char_vertices=ch_boxes)
        
        w_mask  = [1 for _ in range(w_boxes.shape[0])]    # 1 measn readable text
        ch_mask = [1 for _ in range(ch_boxes.shape[0])]
        ch_classes = self.words_to_char_indices_flattened(words)
        score_w, geo_w, ignored_w, _ = get_score_geo(pil_img, w_boxes, w_mask , scale=0.25, length=img_newsize, classes=w_mask)
        score_ch, geo_ch, ignored_ch, class_map = get_score_geo(pil_img, ch_boxes, ch_mask , scale=0.25, length=img_newsize, classes=ch_classes)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])  # ToTensor normalizes between 0 and 1
        pil_img = transform(pil_img)
        w_boxes = torch.Tensor(w_boxes)
        ch_boxes = torch.Tensor(ch_boxes)
        word_indices = torch.LongTensor(self.words_to_indices(words))
        return (pil_img, score_w, geo_w, ignored_w, score_ch, geo_ch, ignored_ch, w_boxes, ch_boxes , word_indices, class_map, path)


def vis(img, word_bbs, char_bbs, txts, word_line_color=(0, 0, 255), char_line_color=(0, 255, 0)):
    '''show word and character bounding boxes on image.
    Input:
        img      :  img to draw bboxes and text to <numpy.ndarray, uint8 , (h,w,3)>
        word_bbs :  word bounding boxes            <numpy.ndarray, (n,8)>
        char_bbs :  char bounding boxes            <numpy.ndarray, (m,8)>
        txts     :  word corresponing to each word_bbs <string list of size n>
    Output:
        img_word_ins: words transcribed img  <numpy.ndarray, uint8 , (h,w,3)>
        img_char_ins: chars transcribed img  <numpy.ndarray, uint8 , (h,w,3)>
    '''
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
    img_word_ins = img.copy()
    img_char_ins = img.copy()
    for txt,word_bbox in zip(txts, word_bbs):
        word_bbox = word_bbox.astype(np.int32)
        # cv2.polylines(img_word_ins, [word_bbox.reshape(4,2)],
        #               True, word_line_color, 2)
        word_bbox = word_bbox.reshape(4,2)
        for i in range(4):
            cv2.line(img_word_ins, tuple(word_bbox[i]), tuple(word_bbox[(i+1)%4]),colors[i])
        cv2.putText(
            img_word_ins,
            '{}'.format(txt),
            (word_bbox[0][0], word_bbox[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            #(word_bbox[0], word_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, word_line_color, 1
        )

    chars = "".join(txts)
    for ch,ch_bbox in zip(chars, char_bbs):
        ch_bbox = ch_bbox.astype(np.int32)
        # cv2.polylines(img_char_ins, [ch_bbox.reshape(4,2)],
        #                 True, char_line_color, 2)
        ch_bbox = ch_bbox.reshape(4,2)
        for i in range(4):
            cv2.line(img_char_ins, tuple(ch_bbox[i]),tuple(ch_bbox[(i+1)%4]),colors[i])
        cv2.putText(
            img_char_ins,
            '{}'.format(ch),
            (ch_bbox[0][0], ch_bbox[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            #(ch_bbox[0], ch_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, char_line_color, 1
        )
    
    return img_word_ins, img_char_ins


def order_points(pnts):
    '''
    initialzie a list of coordinates that will be ordered
    such that the first entry in the list is the top-left,
    the second entry is the top-right, the third is the
    bottom-right, and the fourth is the bottom-left.
    Input:
        pnts: points to be sorted <numpy.ndarray, (8,)>
    Output:
        rect: sorted points <numpy.ndarray, (8,)>
    '''
    x = pnts[::2]
    y = pnts[1::2]
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y) 
    # return the ordered coordinates
    return np.array([xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax])


## TODO vectorize
def detect_out_of_region_bboxes(img_size, word_bbs, char_bbs, txts):
    '''Checks if any of the points of bbox is inside image
    Input:
        img_size : height and width of image       <tuple , (h,w)>
        word_bbs :  word bounding boxes            <numpy.ndarray, (n,8)>
        char_bbs :  char bounding boxes            <numpy.ndarray, (m,8)>
        txts     :  word corresponing to each word_bbs <string list of size n>
    '''
    h,w, = img_size
    p1 = Polygon(np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])).convex_hull
    words = []
    word_bboxes = []
    char_bboxes = []
    if word_bbs.size is 0:
        return word_bbs, char_bbs, txts

    start = 0
    for wbbox, txt in zip(word_bbs,txts):
        p2 = Polygon(wbbox.reshape((4,2))).convex_hull
        inter = p1.intersection(p2)
        end = start + len(txt)
        if p2.area == 0:
            continue            
        if 0 < inter.area / p2.area :  
            #wbbox = list(zip(*inter.exterior.coords.xy))  # extract intersection points
            char_bboxes_of_word = char_bbs[start:end]
            word = ""
            for i,cbbox in enumerate(char_bboxes_of_word):
                p2 = Polygon(cbbox.reshape((4,2))).convex_hull
                inter = p1.intersection(p2)
                if p2.area == 0:
                    continue   
                if 0.5 < inter.area / p2.area:
                    char_bboxes.append(cbbox)                    
                    word += txt[i] # a point is inside image boundaries
            words.append(word)
            word_bboxes.append(wbbox)
        start = end  # start from next word
    if len(char_bboxes)==0:
        return np.empty(0),np.empty(0),[]
    return np.vstack(word_bboxes), np.vstack(char_bboxes), words







if __name__ == '__main__':
    import argparse
    from charnet.modeling.utils import show_img
    from torch.utils.data.sampler import SubsetRandomSampler
    import seaborn as sns; sns.set()


    # reproducability
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)


    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--synthtext', default='/media/end_z820_1/Yeni Birim/DATASETS/SynthText/SynthText.zip', type=str,
                            help='path for synthtext dataset')
        args = parser.parse_args()
        return args

    args = parse_args()
    time1 = time.time()
    dataset = SynthTextDataset(args.synthtext)
    print('| Time taken for data init %.2f' % (time.time() - time1))
    # im, wboxes, cboxes, txts = dataset[0]  # im is PIL ımage
    # img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)  # PIL to cv2
    # img_words, img_chars = vis(img, wboxes, cboxes, txts)
    # cv2.imwrite("res_1.jpg", img_words)
    # cv2.imwrite("res_2.jpg", img_chars)
    # cv2.imwrite("org.jpg", img)

    # img_newsize = 512
    # labels_readable = [1 for _ in range(wboxes.shape[0])]
    # img_new, wboxes = adjust_height(im, wboxes) 
    # img_new, wboxes = rotate_img(img_new, wboxes)
    # img_new, wboxes = crop_img(img_new, wboxes, labels_readable, img_newsize)  


    # score, geo, _ = get_score_geo(img_new, wboxes, labels_readable, scale=0.25, length=img_newsize)
    # img_new =  cv2.cvtColor(np.array(img_new), cv2.COLOR_RGB2BGR)  # PIL to cv2   
    # img_new = cv2.resize(img_new, (128, 128), interpolation=cv2.INTER_LINEAR)
    # score = score.to(torch.float).numpy().transpose(1, 2, 0)  # from C H W to  H W C
    # cv2.imwrite("scored_times_img.jpg", img_new*score)
    # cv2.imwrite("score.jpg", score*255)
    # reshape_geo = geo.unsqueeze(dim=0).permute(1,0,2,3) # / 255.
    # #reshape_geo = gt_geo[0].unsqueeze(dim=0).permute(1,0,2,3) / 255.
    # torchvision.utils.save_image(reshape_geo,"geo_grid.jpg", normalize=True)


    train_size = int(cfg.validation_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=cfg.train_batch_size, shuffle=True, collate_fn=my_collate)

    for im, score_w, geo_w, ignored_w, score_ch, geo_ch, ignored_ch, w_boxes, ch_boxes, words, class_map, paths in train_loader:
        # score_w = score_w.to(torch.float).numpy().transpose(1, 2, 0)  # from C H W to  H W C
        # score_ch = score_ch.to(torch.float).numpy().transpose(1, 2, 0)  # from C H W to  H W C
        print(paths)
        # save colored images 
        torchvision.utils.save_image(im,"img_org_grid.jpg")
        score_w = score_w.type(torch.FloatTensor)
        score_ch = score_ch.type(torch.FloatTensor)
        torchvision.utils.save_image(score_w,"score_w.jpg", normalize=False)  
        torchvision.utils.save_image(score_ch,"score_ch.jpg", normalize=False) 
        for i, geo in enumerate(geo_w):  # 5 channels in eacgeo
            geo = geo.unsqueeze(dim=0).permute(1,0,2,3)  # B=1 C=5 H W  to C=5 B=1 H W
            torchvision.utils.save_image(geo,f"geo_w_{i}_grid.jpg", normalize=True) 
        # save text nontext regions and corrresponding bounding boxes
        torchvision.utils.save_image(score_ch,"score_ch.jpg", normalize=True)  
        for i, geo in enumerate(geo_ch):
            geo = geo.unsqueeze(dim=0).permute(1,0,2,3)
            torchvision.utils.save_image(geo,f"geo_ch_{i}_grid.jpg", normalize=True)  
        # check if char geo intersects with word geo
        mult_geo = geo_ch+geo_w
        for i, geo in enumerate(mult_geo):
            geo = geo.unsqueeze(dim=0).permute(1,0,2,3)
            torchvision.utils.save_image(geo,f"mult_geo_{i}_grid.jpg", normalize=True) 
        torchvision.utils.save_image(score_ch + score_w,"score_ch_w_grid.jpg", normalize=True)
        m = 0
        for img, wboxes, cboxes,wrds, cls_map in zip(im,w_boxes,ch_boxes,words, class_map):
            wrds = dataset.get_words(wrds.numpy())
            img_words, img_chars = vis(img.permute(1,2,0).numpy()*255, wboxes.numpy(), cboxes.numpy(), wrds)
            cv2.imwrite("gt_wrd_bboxes.jpg", img_words) 
            cv2.imwrite("gt_chr_bboxes.jpg", img_chars)


            # img = np.ones((128,128,3))
            # for r in range(128):
            #     for c in range(128):
            #         img[r ,c, :] = rgb_colors[cls_map[0, r, c]]
            # m += 1
            # cv2.imwrite("class_colors_"+str(m)+".jpg", img)

            # ax = sns.heatmap(cls_map[0].numpy(), annot=True, fmt="d")
            # figure = ax.get_figure()    
            # figure.savefig('class_conf.png', dpi=400)
            





    

