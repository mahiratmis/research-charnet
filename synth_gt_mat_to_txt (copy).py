import os
import scipy.io as io
from tqdm import tqdm

gt_mat_path = 'data/SynthText/gt.mat'
im_root = 'data/SynthText/'
txt_root = 'data/SynthText/gt/'

if not os.path.exists(txt_root):
    os.mkdir(txt_root)

print('reading data from {}'.format(gt_mat_path))
gt = io.loadmat(gt_mat_path)
print('Done.')

for i, imname in enumerate(tqdm(gt['imnames'][0])):
    imname = imname[0]
    img_id = os.path.basename(imname)
    im_path = os.path.join(im_root, imname)
    txt_path = os.path.join(txt_root, img_id.replace('jpg', 'txt'))

    if len(gt['wordBB'][0,i].shape) == 2:
        annots = gt['wordBB'][0,i].transpose(1, 0).reshape(-1, 8)
    else:
        annots = gt['wordBB'][0,i].transpose(2, 1, 0).reshape(-1, 8)
    with open(txt_path, 'w') as f:
        f.write(imname + '\n')
        for annot in annots:
            str_write = ','.join(annot.astype(str).tolist())
            f.write(str_write + '\n')

print('End.')

######   SECOND WAY

import scipy.io as sio
import numpy as np
import xml.dom.minidom
import sys
import random
import os

def MatRead(matfile):
    data = sio.loadmat(matfile)

    train_file = open('train.txt', 'w')
    test_file = open('test.txt', 'w')
    
    for i in range(len(data['txt'][0])):
        contents = []
        for val in data['txt'][0][i]:
            v = [x.split("\n") for x in val.strip().split(" ")]
            contents.extend(sum(v, []))
        print >> sys.stderr, "No.{} data".format(i)
        rec = np.array(data['wordBB'][0][i], dtype=np.int32)
        if len(rec.shape) == 3:
            rec = rec.transpose(2,1,0)
        else:
            rec = rec.transpose(1,0)[np.newaxis, :]

        doc = xml.dom.minidom.Document() 
        root = doc.createElement('annotation') 
        doc.appendChild(root) 
        print("start to process {} object".format(len(rec)))
        
        for j in range(len(rec)):
            nodeobject = doc.createElement('object')
            nodecontent = doc.createElement('content')
            nodecontent.appendChild(doc.createTextNode(str(contents[j])))

            nodename = doc.createElement('name')
            nodename.appendChild(doc.createTextNode('text'))

            bndbox = {}
            bndbox['x1'] = rec[j][0][0]
            bndbox['y1'] = rec[j][0][1]
            bndbox['x2'] = rec[j][1][0]
            bndbox['y2'] = rec[j][1][1]
            bndbox['x3'] = rec[j][2][0]
            bndbox['y3'] = rec[j][2][1]
            bndbox['x4'] = rec[j][3][0]
            bndbox['y4'] = rec[j][3][1]
            bndbox['xmin'] = min(bndbox['x1'], bndbox['x2'], bndbox['x3'], bndbox['x4'])
            bndbox['xmax'] = max(bndbox['x1'], bndbox['x2'], bndbox['x3'], bndbox['x4'])
            bndbox['ymin'] = min(bndbox['y1'], bndbox['y2'], bndbox['y3'], bndbox['y4'])
            bndbox['ymax'] = max(bndbox['y1'], bndbox['y2'], bndbox['y3'], bndbox['y4'])

            nodebndbox = doc.createElement('bndbox')
            for k in bndbox.keys():
                nodecoord =  doc.createElement(k)
                nodecoord.appendChild(doc.createTextNode(str(bndbox[k])))
                nodebndbox.appendChild(nodecoord)

            nodeobject.appendChild(nodecontent)
            nodeobject.appendChild(nodename)
            nodeobject.appendChild(nodebndbox)
            root.appendChild(nodeobject)

        filename = data['imnames'][0][i][0].replace('.jpg', '.xml')
        fp = open(filename, 'w')
        doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
        fp.close()
        rad = random.uniform(10,20)
        pwd = os.getcwd()
        img_path = os.path.join(pwd, data['imnames'][0][i][0])
        xml_path = os.path.join(pwd, filename)
        file_line = img_path + " " + xml_path + '\n'
        if rad > 18:
            train_file.write(file_line)
        else:
            test_file.write(file_line)    

    train_file.close()
    test_file.close()