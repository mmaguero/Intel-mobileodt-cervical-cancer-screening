# Test num 001. Marek
# <https://www.kaggle.com/marek3000/test-num-001/code/>

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list
# the files in the input directory


# Any results you write to the current directory are saved as output.
import glob
import cv2
import os
from PIL import ImageFilter, ImageStat, Image, ImageDraw
from datetime import datetime
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from platform import system
from matplotlib import pyplot as pp
from matplotlib import colors as pc
from keras.wrappers.scikit_learn import KerasClassifier
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils import plot_model
from keras.optimizers import SGD

# To Avoid Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

imgSize = 150
prepareData = False
saveNetArchImage = False
NumEpoch = 1
batchSize = 1
percentTrainForValidation = 0.9975
loadPreviousModel = True
pathToPreviousModel = "saved_data/scratch_model_ep00_11-06-2017_11-56.hdf5"
ftModel = "IV3" # IV3/VGG16/RN50 = InceptionV3[Min.139|Def.299]/VGG16[Min.48|Def.224]/ResNet50[Min.197|Rec.224]

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


def create_InceptionV3_model(opt_='adadelta'): #adamax
    # TODO Probar redimensionando las imagenes a 299x299
    model = InceptionV3(include_top=False, weights='imagenet', input_shape=(3, imgSize,
                                                                            imgSize))
    model.compile(optimizer=opt_,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def create_VGG_model(opt_='adadelta'): #adamax
    # TODO Probar redimensionando las imagenes a 299x29

    model = VGG16(include_top=False, weights='imagenet', input_shape=(imgSize,
                                                                            imgSize, 3))
    model.compile(optimizer=opt_,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])   

    return model

# https://keras.io/applications/
# http://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/ 
def create_pretrained_model(baseModel, opt_='adadelta'):
    if (baseModel == "VGG16"):
        myModel = VGG16(weights='imagenet', include_top=False)
    elif (baseModel == "IV3"):
        myModel = InceptionV3(weights='imagenet',include_top=False, input_shape=(3, imgSize, imgSize))
    elif (baseModel == "RN50"):
        myModel = ResNet50(weights='imagenet',include_top=False)

    x = Flatten()(myModel.output)
    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=myModel.input, outputs=output)

    for layer in myModel.layers:
        layer.trainable = False

    model.compile(optimizer=opt_,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def dataPreparation():
    train = glob.glob("../data/train_256_extra/**/*.jpg")

    # train=glob.glob('../input/train/Type_1/*.jpg')[:5] +
    # glob.glob('../input/train/Type_2/*.jpg')[:5] +
    # glob.glob('../input/train/Type_3/*.jpg')[:5]
    print("\nLoading train images...\n" + SEPARATOR)

    if(system().lower() == "windows"):
        train = pd.DataFrame([[p.split('/')[2].split('\\')[1],
                           p.split('/')[2].split('\\')[2], p]
                          for p in train], columns=['type', 'image', 'path'])
    elif(system().lower() == "linux"):
        train = pd.DataFrame([[p.split('/')[3],
                           p.split('/')[4], p]
                          for p in train], columns=['type', 'image', 'path'])

    train = im_stats(train)
    print("\nRemoving bad train images..\n" + SEPARATOR)
    train = train[train['size'] != '0 0'].reset_index(
        drop=True)  # remove bad images

    print("\nNormalizing train images...\n" + SEPARATOR)
    train_data = normalize_image_features(train['path'])
    # train_data = roi(pathtrain)

    print("\nSaving train images...\n" + SEPARATOR)
    np.save('saved_data/train.npy', train_data,
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
    if system().lower() == "windows":
        test = pd.DataFrame([[p.split('/')[2].split('\\')[1], p]
                         for p in test], columns=['image', 'path'])
    elif(system().lower() == "linux"):
        test = pd.DataFrame([[p.split('/')[3], p]
                         for p in test], columns=['image', 'path'])
        
    # [::20] #limit for Kaggle Demo

    print("\nNormalizing test images...\n" + SEPARATOR)
    test_data = normalize_image_features(test['path'])
    # test_data=roi(pathtest)
    print("\nSaving test images...\n" + SEPARATOR)
    np.save('saved_data/test.npy', test_data,
            allow_pickle=True, fix_imports=True)

    test_id = test.image.values
    print("\nSaving test images IDs...\n" + SEPARATOR)
    np.save('saved_data/test_id.npy', test_id,
            allow_pickle=True, fix_imports=True)


def main():
    if (prepareData):
        dataPreparation()

    K.set_image_data_format('channels_first')
    K.set_floatx('float32')

    np.random.seed(17)

    print("\nLoading train data...\n" + SEPARATOR)
    train_data = np.load('saved_data/trainExtra150.npy')
    train_target = np.load('saved_data/trainExtra_target.npy')

    x_train, x_val_train, y_train, y_val_train = train_test_split(
        train_data, train_target, test_size=percentTrainForValidation,
        random_state=17)

    datagen = ImageDataGenerator(rotation_range=0.3, zoom_range=0.3)

    print("\nMaking data augmentation...\n" + SEPARATOR)
    datagen.fit(train_data)

    print("\nCreating model...\n" + SEPARATOR)
    if (loadPreviousModel):
        model = load_model(pathToPreviousModel)
        print("Loaded model from: " + pathToPreviousModel)
        model.summary()
    else:
        model = create_pretrained_model(ftModel)

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    print("\nTraining Set shape (num Instances, RGB chanels, width, height): " + str(
        x_train.shape) + "\nTraining labels: " + str(y_train.shape) + "\n" + SEPARATOR)

    currentDate = datetime.today()
    timeStamp = currentDate.strftime("%d-%m-%Y_%H-%M")

    if (saveNetArchImage):
        plot_model(model, to_file='saved_data/model_' + timeStamp + '.png')

    print("\nFitting model...\n" + SEPARATOR)
    checkPoint = ModelCheckpoint(
        "saved_data/scratch_model_ep{epoch:02d}_" + timeStamp + ".hdf5",
        save_best_only=True)
    # tfBoard = TensorBoard("saved_data/log", histogram_freq=2, write_graph=True, write_images=True,
    # embeddings_freq=2)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batchSize,
                                     shuffle=True),
                        steps_per_epoch=len(x_train), epochs=NumEpoch,
                        validation_data=(x_val_train, y_val_train),
                        callbacks=[checkPoint], verbose=1)

    test_data = np.load('saved_data/test150.npy')
    test_id = np.load('saved_data/test_id.npy')
    print("\nLoaded test data...\n" + SEPARATOR)

    print("\nPredicting with model...\n" + SEPARATOR)
    #pred = model.predict_proba(test_data)
    pred = model.predict(test_data)

    df = pd.DataFrame(pred, columns=['Type_1', 'Type_2', 'Type_3'])
    df['image_name'] = test_id

    df.to_csv("../submission/Test001_Marek_" + timeStamp + ".csv", index=False)


if __name__ == '__main__':
    main()
