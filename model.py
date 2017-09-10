import cv2
from utils import YoloUtils

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
