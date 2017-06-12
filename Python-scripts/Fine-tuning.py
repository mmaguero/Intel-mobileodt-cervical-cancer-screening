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

from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils import plot_model
from keras.applications.inception_v3 import InceptionV3

from image_utils import ImageUtils
from data_augmentation import DataAugmentation as da

# To Avoid Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

imgSize = 150
prepareData = False
useAditional = True
keepAspectRatio = False
useKaggleData = False
saveNetArchImage = True
NumEpoch = 1
batchSize = 32
percentTrainForValidation = 0.7
loadPreviousModel = False
pathToPreviousModel = "saved_data/scratch_model_ep00_11-06-2017_11-56.hdf5"
onlyEvaluate = False
ftModel = "IV3"  # IV3/VGG16/ = InceptionV3[Min.139|Def.299]/VGG16[Min.48|Def.224]
ftApply = True

SEPARATOR = "=============================================================" + \
            "==================="
useCustomPretrainedModels = True

if(useCustomPretrainedModels):

    from pretrained import VGG16
else:
    from keras.applications.vgg16 import VGG16


# https://keras.io/applications/
# http://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/ 
# https://pastebin.com/CWZBeDEb
# Theano como backend: necesita estar a la ultima version (tambien Keras), tomar de los repos de GITHUB
def create_pretrained_model(baseModel, fineTunning, opt_='adadelta'):
    if (baseModel == "VGG16"):
        myModel = VGG16(weights='imagenet', include_top=False, input_shape=(3, imgSize, imgSize))
    elif (baseModel == "IV3"):
        myModel = InceptionV3(weights='imagenet', include_top=False, input_shape=(3, imgSize, imgSize))
    else:
        print("Error: modelo no admitido")
        exit(1)

        '''
    # https://github.com/flyyufelix/cnn_finetune/blob/master/vgg16.py
    if (fineTunning):
        # Truncate and replace softmax layer for transfer learning
        myModel.layers.pop()
        myModel.outputs = [myModel.layers[-1].output]
        myModel.layers[-1].outbound_nodes = []
'''
    x = Flatten()(myModel.output)
    x = Dense(256, activation='relu')(x)
    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=myModel.input, outputs=output)

    for i, layer in enumerate(myModel.layers):
        print(i, layer.name)


    for layer in myModel.layers:
        layer.trainable = False


    model.compile(optimizer=opt_,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def create_fine_tunning_model(opt_='adadelta'):
    model = VGG16(weights='imagenet', include_top=False, input_shape=(3, imgSize, imgSize))

    return model


def evaluateModel(model, testData, testLabels):
    score = model.evaluate(testData, testLabels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def main():
    if (prepareData):
        imgUtils = ImageUtils(imgSize, useAditional=useAditional, keepAspectRatio=keepAspectRatio,
                              useKaggleData=useKaggleData)
        imgUtils.dataPreparation()()

    K.set_image_data_format('channels_first')
    K.set_floatx('float32')

    np.random.seed(17)

    print("\nLoading train data...\n" + SEPARATOR)
    if (keepAspectRatio):
        if (useAditional):
            train_data = np.load('saved_data/trainExtra' + str(imgSize) + '_OrigAspectRatio.npy')
            train_target = np.load('saved_data/trainExtra_target.npy')
        else:
            train_data = np.load('saved_data/train' + str(imgSize) + '_OrigAspectRatio.npy')
            train_target = np.load('saved_data/train_target.npy')
    else:

        if (useAditional):
            train_data = np.load('saved_data/trainExtra' + str(imgSize) + '.npy')
            train_target = np.load('saved_data/trainExtra_target.npy')
        else:
            train_data = np.load('saved_data/train' + str(imgSize) + '.npy')
            train_target = np.load('saved_data/train_target.npy')

    x_train, x_val_train, y_train, y_val_train = train_test_split(
        train_data, train_target, test_size=percentTrainForValidation,
        random_state=17)

    print("\nMaking data augmentation...\n" + SEPARATOR)
    datagen = da.prepareDataAugmentation(train_data=train_data)

    print("\nCreating model...\n" + SEPARATOR)
    if (loadPreviousModel):
        model = load_model(pathToPreviousModel)
        print("Loaded model from: " + pathToPreviousModel)
    else:
        model = create_pretrained_model(ftModel, ftApply)

    model.summary()

    print("\nTraining Set shape (num Instances, RGB chanels, width, height): " + str(
        x_train.shape) + "\nTraining labels: " + str(y_train.shape) + "\nValidating set shape: " + str(
        x_val_train.shape) + "\nValidating set labels: " + str(
        y_val_train.shape) + "\n" + SEPARATOR)

    currentDate = datetime.today()
    timeStamp = currentDate.strftime("%d-%m-%Y_%H-%M")

    if (saveNetArchImage):
        plot_model(model, to_file='saved_data/'+ftModel+'_' + timeStamp + '.png')

    if (onlyEvaluate):

        print("\nEvaluating Model...\n" + SEPARATOR)
        evaluateModel(model, x_val_train, y_val_train)

    else:
        print("\nFitting model...\n" + SEPARATOR)
        checkPoint = ModelCheckpoint(
            "saved_data/" + ftModel + "_ep{epoch:02d}_" + timeStamp + ".hdf5",
            save_best_only=True)

        if ftApply == False:
            model.fit_generator(datagen.flow(x_train, y_train, batch_size=batchSize,
                                             shuffle=True),
                                steps_per_epoch=len(x_train), epochs=NumEpoch,
                                validation_data=(x_val_train, y_val_train),
                                callbacks=[checkPoint])#, verbose=1)
        else:
            model.fit_generator(datagen.flow(x_train, y_train, batch_size=batchSize,
                                             shuffle=True),
                                steps_per_epoch=len(x_train), epochs=NumEpoch,
                                validation_data=(x_val_train, y_val_train))  # , verbose=1)

        if (ftApply):
            # Set the first layers to non-trainable (weights will not be updated)
            # Set the last block to trainable
            print("\nFine-tunning model...\n" + SEPARATOR)
            if (ftModel == "VGG16"):
                for layer in model.layers[:15]:
                    layer.trainable = False
                for layer in model.layers[15:]:
                    layer.trainable = True
            elif (ftModel =="IV3"):
                for layer in model.layers[:277]:
                    layer.trainable = False
                for layer in model.layers[277:]:
                    layer.trainable = True

            # compile the model with a SGD/momentum optimizer
            # and a very slow learning rate.
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                          metrics=['accuracy'])

            model.fit_generator(datagen.flow(x_train, y_train, batch_size=batchSize,
                                             shuffle=True),
                                steps_per_epoch=len(x_train), epochs=NumEpoch,
                                validation_data=(x_val_train, y_val_train),
                                callbacks=[checkPoint])  # , verbose=1)



    print("\nLoading test data...\n" + SEPARATOR)

    if (keepAspectRatio):
        test_data = np.load('saved_data/test' + str(imgSize) + '_OrigAspectRatio.npy')
        test_id = np.load('saved_data/test_id.npy')
    else:
        test_data = np.load('saved_data/test' + str(imgSize) + '.npy')
        test_id = np.load('saved_data/test_id.npy')

    print("\nPredicting with model...\n" + SEPARATOR)
    pred = model.predict(test_data, verbose=1)

    df = pd.DataFrame(pred, columns=['Type_1', 'Type_2', 'Type_3'])
    df['image_name'] = test_id

    df.to_csv("../submission/" + ftModel + "_" + timeStamp + ".csv", index=False)


if __name__ == '__main__':
    main()
