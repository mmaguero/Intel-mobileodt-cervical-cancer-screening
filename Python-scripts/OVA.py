# Test num 001. Marek
# <https://www.kaggle.com/marek3000/test-num-001/code/>

# Any results you write to the current directory are saved as output.
from datetime import datetime

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

from data_augmentation import DataAugmentation as da
from image_utils import ImageUtils

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list
# the files in the input directory

# To Avoid Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

imgSize = 64
prepareData = False
useAditional = True
keepAspectRatio = False
useKaggleData = False
saveNetArchImage = False
NumEpoch = 1
batchSize = 32
percentTrainForValidation = 0.2
loadPreviousModel = False
pathToPreviousModel = ["saved_data/scratch_model_ep05_10-06-2017_22-08.hdf5",
                       "saved_data/scratch_model_ep05_10-06-2017_22-08.hdf5",
                       "saved_data/scratch_model_ep05_10-06-2017_22-08.hdf5"]
onlyEvaluate = False

SEPARATOR = "=============================================================" + \
            "==================="


def create_model(opt_='adadelta'):
    model = Sequential()
    model.add(Conv2D(4, (3, 3), activation='relu', input_shape=(3, imgSize,
                                                                imgSize),
                     data_format="channels_first"))  # input_shape=(3,64,64)
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3),
                           data_format="channels_first"))
    model.add(Conv2D(8, (3, 3), activation='relu',
                     data_format="channels_first"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3),
                           data_format="channels_first"))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(12, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=opt_,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def evaluateModel(model, testData, testLabels):
    score = model.evaluate(testData, testLabels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def main():
    if (prepareData):
        imgUtils = ImageUtils(imgSize, useAditional=useAditional, keepAspectRatio=keepAspectRatio,
                              useKaggleData=useKaggleData)
        imgUtils.dataPreparationOVA()

    K.set_image_data_format('channels_first')
    K.set_floatx('float32')

    np.random.seed(17)

    print("\nLoading train data...\n" + SEPARATOR)

    train_target = []
    if (keepAspectRatio):
        if (useAditional):
            train_data = np.load('saved_data/trainExtra' + str(imgSize) + '_OrigAspectRatio.npy')

            train_target.append(np.load('saved_data/trainExtraOVA1_target.npy'))
            train_target.append(np.load('saved_data/trainExtraOVA2_target.npy'))
            train_target.append(np.load('saved_data/trainExtraOVA3_target.npy'))

        else:
            train_data = np.load('saved_data/train' + str(imgSize) + '_OrigAspectRatio.npy')

            train_target.append(np.load('saved_data/train_targetOVA1.npy'))
            train_target.append(np.load('saved_data/train_targetOVA2.npy'))
            train_target.append(np.load('saved_data/train_targetOVA3.npy'))
    else:

        if (useAditional):
            train_data = np.load('saved_data/trainExtra' + str(imgSize) + '.npy')

            train_target.append(np.load('saved_data/trainExtraOVA1_target.npy'))
            train_target.append(np.load('saved_data/trainExtraOVA2_target.npy'))
            train_target.append(np.load('saved_data/trainExtraOVA3_target.npy'))
        else:
            train_data = np.load('saved_data/train' + str(imgSize) + '.npy')

            train_target.append(np.load('saved_data/train_targetOVA1.npy'))
            train_target.append(np.load('saved_data/train_targetOVA2.npy'))
            train_target.append(np.load('saved_data/train_targetOVA3.npy'))

    print("\nMaking data augmentation...\n" + SEPARATOR)
    datagen = da.prepareDataAugmentation(train_data=train_data)

    model = []
    currentDate = datetime.today()
    timeStamp = currentDate.strftime("%d-%m-%Y_%H-%M")

    for i in range(len(train_target)):

        x_train, x_val_train, y_train, y_val_train = train_test_split(
            train_data, train_target[i], test_size=percentTrainForValidation,
            random_state=17)

        print("\nCreating model " + str(i+1) + "...\n" + SEPARATOR)
        if (loadPreviousModel):
            model.append(load_model(pathToPreviousModel[i]))
            print("Loaded model from: " + pathToPreviousModel[i])
            model[i].summary()
        else:
            model.append(create_model())

        print("\nTraining Set shape (num Instances, RGB chanels, width, height): " + str(
            x_train.shape) + "\nTraining labels: " + str(y_train.shape) + "\nValidating set shape: " + str(
            x_val_train.shape) + "\nValidating set labels: " + str(
            y_val_train.shape) + "\n" + SEPARATOR)

        if (saveNetArchImage):
            plot_model(model[i], to_file='saved_data/model_' + timeStamp + '.png')

        if (onlyEvaluate):

            print("\nEvaluating Model " + str(i+1) + "...\n" + SEPARATOR)
            evaluateModel(model[i], x_val_train, y_val_train)

        else:
            print("\nFitting model " + str(i+1) + "...\n" + SEPARATOR)
            checkPoint = ModelCheckpoint(
                "saved_data/OVA_model" + str(i+1) + "_ep{epoch:02d}_" + timeStamp + ".hdf5",
                save_best_only=True)

            model[i].fit_generator(datagen.flow(x_train, y_train, batch_size=batchSize,
                                                shuffle=True),
                                   steps_per_epoch=10, epochs=NumEpoch,
                                   validation_data=(x_val_train, y_val_train),
                                   callbacks=[checkPoint])  # , verbose=2)

    print("\nLoading test data...\n" + SEPARATOR)

    if (keepAspectRatio):
        test_data = np.load('saved_data/test' + str(imgSize) + '_OrigAspectRatio.npy')
        test_id = np.load('saved_data/test_id.npy')
    else:
        test_data = np.load('saved_data/test' + str(imgSize) + '.npy')
        test_id = np.load('saved_data/test_id.npy')

    pred = []
    for i in range(len(model)):
        print("\nPredicting with model " + str(i+1) + "...\n" + SEPARATOR)
        pred.append(model[i].predict_proba(test_data))

    print(pred[0][:, 1])
    print(pred[1][:, 1])
    print(pred[2][:, 1])

    df = pd.DataFrame([pred[0][:, 1], pred[1][:, 1], pred[2][:, 1]], columns=['Type_1', 'Type_2', 'Type_3'])
    df['image_name'] = test_id

    df.to_csv("../submission/OVA_" + timeStamp + ".csv", index=False)


if __name__ == '__main__':
    main()
