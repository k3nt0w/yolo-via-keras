from keras.models import Sequential, Model
from keras.layers import Input, BatchNormalization, Flatten, Dense, Reshape, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from vgg19 import VGG19
from keras import backend as K

import cv2
from utils import YoloUtils

'''
img_path = '/Users/kento_watanabe/MATHGRAM/yolo-via-keras/VOCdevkit/VOC2012/JPEGImages/2008_000002.jpg'
label_path = 'VOCdevkit/VOC2012/labels/2008_000002.txt'
def gen_data(img_paths, label_paths):
    utils = YoloUtils(480, 7)
    for img_path, label_path in zip([img_paths], [label_paths]):
        print(img_path, label_path)
        lines = []
        with open(label_path, 'r') as f:
            lines.append(list(map(float, f.readline().split())))
        for line in lines:
            x = cv2.imread(img_path)
            y = utils.make_train_map(line)
            yield(x, y)

gen_data(img_path, label_path)
'''

def vgg_yolo():
    vgg19 = VGG19(include_top=False, weights='imagenet',
                  input_tensor=None, input_shape=(416,416,3))
    ip = Input(shape=(416, 416, 3))
    # Block1
    h = vgg19.layers[1](ip)
    h = BatchNormalization()(h)
    h = LeakyReLU(alpha=0.1)(h)
    h = vgg19.layers[2](h)
    h = BatchNormalization()(h)
    h = LeakyReLU(alpha=0.1)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    # Block2
    for i in range(4, 6):
        h = vgg19.layers[i](h)
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.1)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    # Block3
    for i in range(7, 11):
        h = vgg19.layers[i](h)
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.1)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    # Block4
    for i in range(12, 16):
        h = vgg19.layers[i](h)
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.1)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    # Block5
    for i in range(17, 21):
        h = vgg19.layers[i](h)
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.1)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    # Block6
    for _ in range(0,2):
        h = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False)(h)
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.1)(h)
    # Block7
    h = Conv2D(125, (1, 1), strides=(1, 1), kernel_initializer='he_normal')(h)
    h = Activation('linear')(h)
    h = Reshape((13, 13, 5, 25))(h)

    return Model(ip, h)

def yolo():
    ip = Input(shape=(416, 416, 3))
    h = Conv2D(16, (3, 3), strides=(1, 1), padding='same', use_bias=False)(ip)
    h = BatchNormalization()(h)
    h = LeakyReLU(alpha=0.1)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)

    for i in range(0, 4):
        h = Conv2D(32*(2**i), (3, 3), strides=(1, 1), padding='same', use_bias=False)(h)
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.1)(h)
        h = MaxPool2D(pool_size=(2, 2))(h)

    h = Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False)(h)
    h = BatchNormalization()(h)
    h = LeakyReLU(alpha=0.1)(h)
    h = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')(h)

    for _ in range(0,2):
        h = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False)(h)
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.1)(h)

    h = Conv2D(125, (1, 1), strides=(1, 1), kernel_initializer='he_normal')(h)
    h = Activation('linear')(h)
    h = Reshape((13, 13, 5, 25))(h)

    return Model(ip, h)

if __name__ == "__main__":
    from keras.utils import plot_model
    #model = yolo()
    model = vgg_yolo()
    plot_model(model, to_file='model.png', show_shapes=True)
    model.summary()
