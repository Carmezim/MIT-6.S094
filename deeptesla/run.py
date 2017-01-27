#!/usr/bin/env python 
from __future__ import division

import tensorflow as tf
import model
import cv2
import subprocess as sp
import itertools
import params
import sys
import os
import preprocess
import visualize
import time

import local_common as cm

sess = tf.InteractiveSession()
saver = tf.train.Saver()
model_name = 'model.ckpt'
model_path = cm.jn(params.save_dir, model_name)
saver.restore(sess, model_path)

epoch_ids = sorted(list(set(itertools.chain(*params.epochs.values()))))

for epoch_id in epoch_ids:
    print '---------- processing video for epoch {} ----------'.format(epoch_id)
    vid_path = cm.jn(params.data_dir, 'epoch{:0>2}_front.mkv'.format(epoch_id))
    assert os.path.isfile(vid_path)
    frame_count = cm.frame_count(vid_path)
    cap = cv2.VideoCapture(vid_path)

    machine_steering = []

    print 'performing inference...'
    time_start = time.time()
    for frame_id in xrange(frame_count):
        ret, img = cap.read()
        assert ret

        img = preprocess.preprocess(img)
        deg = model.y.eval(feed_dict={model.x: [img], model.keep_prob: 1.0})[0][0]
        machine_steering.append(deg)

    cap.release()

    fps = frame_count / (time.time() - time_start)
    
    print 'completed inference, total frames: {}, average fps: {} Hz'.format(frame_count, round(fps, 1))
    
    print 'performing visualization...'
    visualize.visualize(epoch_id, machine_steering, params.out_dir,
                        verbose=True, frame_count_limit=None)
    
    
