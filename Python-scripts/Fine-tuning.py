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
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
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
batchSize = 16
percentTrainForValidation = 0.3
loadPreviousModel = False
pathToPreviousModel = "saved_data/scratch_model_ep00_11-06-2017_11-56.hdf5"
onlyEvaluate = False
ftModel = "IV3" # IV3/VGG16/RN50 = InceptionV3[Min.139|Def.299]/VGG16[Min.48|Def.224]/ResNet50[Min.197|Rec.224]
ftApply = False

SEPARATOR = "=============================================================" + \
            "==================="

'''
def VGG16(weights='imagenet', include_top=False, input_shape=None):
    img_input = Input(shape=input_shape)
    inputs = img_input
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Create model.
    model = Model(inputs, x, name='vgg16')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')


    return model

'''
# https://keras.io/applications/
# http://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/ 
# https://pastebin.com/CWZBeDEb
# Theano como backend: necesita estar a la ultima version (tambien Keras), tomar de los repos de GITHUB
def create_pretrained_model(baseModel, fineTunning, opt_='adadelta'):
    if (baseModel == "VGG16"):
        myModel = VGG16(weights='imagenet', include_top=False, input_shape=(3, imgSize, imgSize))
    elif (baseModel == "IV3"):
        myModel = InceptionV3(weights='imagenet', include_top=False, input_shape=(3, imgSize, imgSize))
    elif (baseModel == "RN50"):
        myModel = ResNet50(weights='imagenet', include_top=False, input_shape=(3, imgSize, imgSize))

    # https://github.com/flyyufelix/cnn_finetune/blob/master/vgg16.py
    if (fineTunning):
        # Truncate and replace softmax layer for transfer learning
        myModel.layers.pop()
        myModel.outputs = [myModel.layers[-1].output]
        myModel.layers[-1].outbound_nodes = []

    x = Flatten()(myModel.output)
    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=myModel.input, outputs=output)

    if (fineTunning):
        #Set the first layers to non-trainable (weights will not be updated)
        for layer in myModel.layers[:25]:
            layer.trainable = False
    else:
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

    datagen = ImageDataGenerator(rotation_range=0.3, zoom_range=0.3)

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
        x_train.shape) + "\nTraining labels: " + str(y_train.shape) + "\n" + SEPARATOR)

    currentDate = datetime.today()
    timeStamp = currentDate.strftime("%d-%m-%Y_%H-%M")

    if (saveNetArchImage):
        plot_model(model, to_file='saved_data/model_' + timeStamp + '.png')

    if (onlyEvaluate):

        print("\nEvaluating Model...\n" + SEPARATOR)
        evaluateModel(model, x_val_train, y_val_train)

    else:
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

    print("\nLoading test data...\n" + SEPARATOR)

    if (keepAspectRatio):
        test_data = np.load('saved_data/test' + str(imgSize) + '_OrigAspectRatio.npy')
        test_id = np.load('saved_data/test_id.npy')
    else:
        test_data = np.load('saved_data/test' + str(imgSize) + '.npy')
        test_id = np.load('saved_data/test_id.npy')

    print("\nPredicting with model...\n" + SEPARATOR)
    #pred = model.predict_proba(test_data)
    pred = model.predict(test_data, verbose=1)

    df = pd.DataFrame(pred, columns=['Type_1', 'Type_2', 'Type_3'])
    df['image_name'] = test_id

    df.to_csv("../submission/Fine_Tuning_" + timeStamp + ".csv", index=False)


if __name__ == '__main__':
    main()
