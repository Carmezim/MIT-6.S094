#!/usr/bin/env python 
from __future__ import division

import os
from collections import OrderedDict

batch_size = 100
save_dir = os.path.abspath('models')
training_steps = 2000
img_height = 66
img_width = 200
img_channels = 3
write_summary = True

# change this to the directory that contains the source videos
data_dir = os.path.abspath('./epochs')
out_dir = os.path.abspath('./output')
shuffle_training = True

assert os.path.isdir(data_dir)
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

epochs = OrderedDict()
# epochs['train'] = range(1, 11)
# epochs['val'] = range(1, 11)
epochs['train'] = [3, 4, 5, 6, 8]
epochs['val'] = [1, 2, 7, 9, 10]

