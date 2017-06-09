# Test num 001. Marek
# <https://www.kaggle.com/marek3000/test-num-001/code/>

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list
# the files in the input directory


# Any results you write to the current directory are saved as output.
from PIL import ImageFilter, ImageStat, Image, ImageDraw
from datetime import datetime
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import glob
import cv2
from matplotlib import pyplot as pp
from matplotlib import colors as pc
import os
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils import plot_model

# To Avoid Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

imgSize = 64

SEPARATOR = "=============================================================" + \
            "==================="


def im_multi(path):
    try:
        im_stats_im_ = Image.open(path)
        return [path, {'size': im_stats_im_.size}]
    except:
        print(path)
        return [path, {'size': [0, 0]}]


#########
##########
'''
types = ['Type_1','Type_2','Type_3']
pathtrain = "../input/train"
pathtest='../input/test'
'''


# loop to extract the ROI in the image and create an additional image with only
# the ROI highlighted


def roi(pathtrain):
    for typ in ['Type_1', 'Type_2', 'Type_3']:
        for img in os.listdir(pathtrain + '/' + typ):
            image = pathtrain + '/' + typ + '/' + img
            # os.chdir(pathtrain + '/' + typ + '/')
            ii = cv2.imread(image)
            # cv2.imshow('image', ii[:, :, 1])
            # cv2.waitKey(0)
            b, g, r = cv2.split(ii)
            rgb_img = cv2.merge([r, g, b])
            rgb_img1 = pc.rgb_to_hsv(rgb_img)
            indices = np.where(rgb_img1[:, :, 0] < 0.7)
            rgb_img1[:, :, 0][indices] = 0
            rgb_img1[:, :, 1][indices] = 0
            rgb_img1[:, :, 2][indices] = 0
            rgb_img1 = pc.hsv_to_rgb(rgb_img1).astype(np.uint8)
            pp.imsave(fname="train_256_roi/" + img.split('.')[0] + '_trans.jpg',
                      arr=rgb_img1)
    return fname


def im_stats(im_stats_df):
    im_stats_d = {}
    p = Pool(cpu_count())
    ret = p.map(im_multi, im_stats_df['path'])
    for i in range(len(ret)):
        im_stats_d[ret[i][0]] = ret[i][1]
    im_stats_df['size'] = im_stats_df['path'].map(
        lambda x: ' '.join(str(s) for s in im_stats_d[x]['size']))

    p.close()
    return im_stats_df


def get_im_cv2(path):
    img = cv2.imread(path)
    # use cv2.resize(img, (64, 64), cv2.INTER_LINEAR)
    resized = cv2.resize(img, (imgSize, imgSize), cv2.INTER_LINEAR)
    return [path, resized]


def normalize_image_features(paths):
    imf_d = {}
    p = Pool(cpu_count())
    ret = p.map(get_im_cv2, paths)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    fdata = np.array(fdata, dtype=np.uint8)
    fdata = fdata.transpose((0, 3, 1, 2))
    fdata = fdata.astype('float32')
    fdata = fdata / 255
    p.close()
    return fdata



def main():
    train = glob.glob("../data/train_256_extra/**/*.jpg")

    # train=glob.glob('../input/train/Type_1/*.jpg')[:5] +
    # glob.glob('../input/train/Type_2/*.jpg')[:5] +
    # glob.glob('../input/train/Type_3/*.jpg')[:5]
    print("\nLoading train images...\n" + SEPARATOR)

    train = pd.DataFrame([[p.split('/')[2].split('\\')[1],
                           p.split('/')[2].split('\\')[2], p]
                          for p in train], columns=['type', 'image', 'path'])

    train = im_stats(train)
    print("\nRemoving bad train images..\n" + SEPARATOR)
    train = train[train['size'] != '0 0'].reset_index(
        drop=True)  # remove bad images

    print("\nNormalizing train images...\n" + SEPARATOR)
    train_data = normalize_image_features(train['path'])
    # train_data = roi(pathtrain)

    print("\nSaving train images...\n" + SEPARATOR)
    np.save('saved_data/train'+str(imgSize)+'.npy', train_data,
            allow_pickle=True, fix_imports=True)

    print("\nGetting train images labels...\n" + SEPARATOR)
    le = LabelEncoder()
    train_target = le.fit_transform(train['type'].values)
    # FIXME da error
    print("\nClases: " + le.classes_ + "\n" +
          SEPARATOR)  # in case not 1 to 3 order
    print("\nSaving train images labels...\n" + SEPARATOR)
    np.save('saved_data/train_target.npy', train_target,
            allow_pickle=True, fix_imports=True)

    test = glob.glob("../data/test_256/*.jpg")
    print("\nLoading test images...\n" + SEPARATOR)
    test = pd.DataFrame([[p.split('/')[2].split('\\')[1], p]
                         for p in test], columns=['image', 'path'])
    # [::20] #limit for Kaggle Demo

    print("\nNormalizing test images...\n" + SEPARATOR)
    test_data = normalize_image_features(test['path'])
    # test_data=roi(pathtest)
    print("\nSaving test images...\n" + SEPARATOR)
    np.save('saved_data/test'+str(imgSize)+'.npy', test_data,
            allow_pickle=True, fix_imports=True)

    test_id = test.image.values
    print("\nSaving test images IDs...\n" + SEPARATOR)
    np.save('saved_data/test_id.npy', test_id,
            allow_pickle=True, fix_imports=True)


if __name__ == '__main__':
    main()
