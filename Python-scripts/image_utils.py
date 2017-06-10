import glob
import os
from multiprocessing import Pool, cpu_count
from platform import system
import cv2
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# Any results you write to the current directory are saved as output.
from PIL import Image
from matplotlib import colors as pc
from matplotlib import pyplot as pp
from sklearn.preprocessing import LabelEncoder


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list
# the files in the input directory


class ImageUtils:
    SEPARATOR = "=============================================================" + \
                "==================="

    def __init__(self, imgSize, useAditional):
        self.imgSize = imgSize
        self.useAditional = useAditional

    def im_multi(self, path):
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


    def roi(self, pathtrain):
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

    def im_stats(self, im_stats_df):
        im_stats_d = {}
        p = Pool(cpu_count())
        ret = p.map(self.im_multi, im_stats_df['path'])
        for i in range(len(ret)):
            im_stats_d[ret[i][0]] = ret[i][1]
        im_stats_df['size'] = im_stats_df['path'].map(
            lambda x: ' '.join(str(s) for s in im_stats_d[x]['size']))

        p.close()
        return im_stats_df

    def get_im_cv2(self, path):
        img = cv2.imread(path)
        # use cv2.resize(img, (64, 64), cv2.INTER_LINEAR)
        resized = cv2.resize(img, (self.imgSize, self.imgSize), cv2.INTER_LINEAR)  # TODO mirar el modo de redimension
        return [path, resized]

    def normalize_image_features(self, paths):
        imf_d = {}
        p = Pool(cpu_count())
        ret = p.map(self.get_im_cv2, paths)
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

    def dataPreparation(self):
        if (self.useAditional):
            train = glob.glob("../data/train_256_extra/**/*.jpg")
        else:
            train = glob.glob("../data/train_256/**/*.jpg")

        # train=glob.glob('../input/train/Type_1/*.jpg')[:5] +
        # glob.glob('../input/train/Type_2/*.jpg')[:5] +
        # glob.glob('../input/train/Type_3/*.jpg')[:5]
        print("\nLoading train images...\n" + self.SEPARATOR)

        if (system().lower() == "windows"):
            train = pd.DataFrame([[p.split('/')[2].split('\\')[1],
                                   p.split('/')[2].split('\\')[2], p]
                                  for p in train], columns=['type', 'image', 'path'])
        elif (system().lower() == "linux"):
            train = pd.DataFrame([[p.split('/')[3],
                                   p.split('/')[4], p]
                                  for p in train], columns=['type', 'image', 'path'])

        train = self.im_stats(train)
        print("\nRemoving bad train images..\n" + self.SEPARATOR)
        train = train[train['size'] != '0 0'].reset_index(
            drop=True)  # remove bad images

        print("\nNormalizing train images...\n" + self.SEPARATOR)
        train_data = self.normalize_image_features(train['path'])
        # train_data = roi(pathtrain)

        print("\nSaving train images...\n" + self.SEPARATOR)
        if (self.useAditional):
            np.save('saved_data/trainExtra' + str(self.imgSize) + '.npy', train_data,
                    allow_pickle=True, fix_imports=True)
        else:
            np.save('saved_data/train' + str(self.imgSize) + '.npy', train_data,
                    allow_pickle=True, fix_imports=True)

        print("\nGetting train images labels...\n" + self.SEPARATOR)
        le = LabelEncoder()
        train_target = le.fit_transform(train['type'].values)
        # FIXME da error
        print("\nClases: " + str(le.classes_) + "\n" +
              self.SEPARATOR)  # in case not 1 to 3 order
        print("\nSaving train images labels...\n" + self.SEPARATOR)

        if (self.useAditional):
            np.save('saved_data/trainExtra_target.npy', train_target,
                    allow_pickle=True, fix_imports=True)
        else:
            np.save('saved_data/train_target.npy', train_target,
                    allow_pickle=True, fix_imports=True)

        test = glob.glob("../data/test_256/*.jpg")
        print("\nLoading test images...\n" + self.SEPARATOR)
        if system().lower() == "windows":
            test = pd.DataFrame([[p.split('/')[2].split('\\')[1], p]
                                 for p in test], columns=['image', 'path'])
        elif (system().lower() == "linux"):
            test = pd.DataFrame([[p.split('/')[3], p]
                                 for p in test], columns=['image', 'path'])
        # [::20] #limit for Kaggle Demo

        print("\nNormalizing test images...\n" + self.SEPARATOR)
        test_data = self.normalize_image_features(test['path'])
        # test_data=roi(pathtest)
        print("\nSaving test images...\n" + self.SEPARATOR)

        np.save('saved_data/test' + str(self.imgSize) + '.npy', train_data,
                allow_pickle=True, fix_imports=True)

        test_id = test.image.values
        print("\nSaving test images IDs...\n" + self.SEPARATOR)
        np.save('saved_data/test_id.npy', test_id,
                allow_pickle=True, fix_imports=True)
