from keras.models import Sequential
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import os

from utils import *
from model import yolo, vgg_yolo
from loss import yolo_loss


LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
          'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
          'dog', 'horse', 'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'tvmonitor']
COLORS = [(43,206,72),(255,204,153),(128,128,128),(148,255,181),(143,124,0),
          (157,204,0),(194,0,136),(0,51,128),(255,164,5),(255,168,187),
          (66,102,0),(255,0,16),(94,241,242),(0,153,143),(224,255,102),
          (116,10,255),(153,0,0),(255,255,128),(255,255,0),(255,80,5)]

ann_dir = 'VOCdevkit/VOC2012/Annotations/'
img_dir = 'VOCdevkit/VOC2012/JPEGImages/'

#model = yolo()
model = vgg_yolo()

# Randomize weights of the last layer
layer = model.layers[-3] # the last convolutional layer
weights = layer.get_weights()

new_kernel = np.random.normal(size=weights[0].shape)/(13*13)
new_bias = np.random.normal(size=weights[1].shape)/(13*13)

layer.set_weights([new_kernel, new_bias])


early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
checkpoint = ModelCheckpoint('weights.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)

tb_counter  = max([int(num) for num in os.listdir('logs/')] or [0]) + 1
tensorboard = TensorBoard(log_dir='logs/' + str(tb_counter), histogram_freq=0, write_graph=True, write_images=False)

sgd = SGD(lr=0.00001, decay=0.0005, momentum=0.9)

all_img = parse_annotation(ann_dir)
model.compile(loss=yolo_loss, optimizer=sgd)
model.fit_generator(data_gen(all_img, 8),
                    int(len(all_img)/8),
                    epochs = 100,
                    callbacks = [early_stop, checkpoint, tensorboard],
                    max_q_size = 3)
