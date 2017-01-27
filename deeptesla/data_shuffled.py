#!/usr/bin/env python
from __future__ import division

import random
import os
import sys
from collections import OrderedDict
import cv2
import params
import preprocess

import local_common as cm

################ parameters ###############
data_dir = params.data_dir
epochs = params.epochs
img_height = params.img_height
img_width = params.img_width
img_channels = params.img_channels

purposes = ['train', 'val']
imgs = OrderedDict()
wheels = OrderedDict()
for purpose in purposes:
    imgs[purpose] = []
    wheels[purpose] = []

# load all preprocessed training images into memory
def load_imgs():
    global imgs
    global wheels

    for p in purposes:
        for epoch_id in epochs[p]:
            print 'processing and loading "{}" epoch {} into memory, current num of imgs is {}...'.format(
                p, epoch_id, len(imgs[p]))

            vid_path = cm.jn(data_dir, 'epoch{:0>2}_front.mkv'.format(epoch_id))
            assert os.path.isfile(vid_path)
            frame_count = cm.frame_count(vid_path)
            cap = cv2.VideoCapture(vid_path)

            csv_path = cm.jn(data_dir, 'epoch{:0>2}_steering.csv'.format(epoch_id))
            assert os.path.isfile(csv_path)
            rows = cm.fetch_csv_data(csv_path)
            assert frame_count == len(rows)
            yy = [[float(row['wheel'])] for row in rows]

            while True:
                ret, img = cap.read()
                if not ret:
                    break

                img = preprocess.preprocess(img)
                imgs[p].append(img)

            wheels[p].extend(yy)
            assert len(imgs[p]) == len(wheels[p])

            cap.release()

def load_batch(purpose):
    p = purpose
    assert len(imgs[p]) == len(wheels[p])
    n = len(imgs[p])
    assert n > 0

    ii = random.sample(xrange(0, n), params.batch_size)
    assert len(ii) == params.batch_size

    xx, yy = [], []
    for i in ii:
        xx.append(imgs[p][i])
        yy.append(wheels[p][i])

    return xx, yy
    
if __name__ == '__main__':
    load_imgs()

    load_batch()
