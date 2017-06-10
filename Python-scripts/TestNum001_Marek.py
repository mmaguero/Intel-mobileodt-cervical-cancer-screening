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
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

from image_utils import ImageUtils

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list
# the files in the input directory

# To Avoid Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

imgSize = 64
prepareData = False
useAditional = True
saveNetArchImage = False
NumEpoch = 1
batchSize = 32
percentTrainForValidation = 0.2
loadPreviousModel = True
pathToPreviousModel = "saved_data/scratch_model_04_09-06-2017_12-26.hdf5"
onlyEvaluate = False

SEPARATOR = "=============================================================" + \
            "==================="



def create_model(opt_='adamax'):
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
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=opt_,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def evaluateModel(model, testData, testLabels):
    score = model.evaluate(testData, testLabels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def main():
    if (prepareData):
        imgUtils = ImageUtils(imgSize, useAditional)
        imgUtils.dataPreparation()

    K.set_image_data_format('channels_first')
    K.set_floatx('float32')

    np.random.seed(17)

    print("\nLoading train data...\n" + SEPARATOR)

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
    datagen.fit(train_data)

    print("\nCreating model...\n" + SEPARATOR)
    if(loadPreviousModel):
        model = load_model(pathToPreviousModel)
        print("Loaded model from: "+pathToPreviousModel)
        model.summary()
    else:
        model = create_model()

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
                            callbacks=[checkPoint], verbose=2)

    print("\nLoading test data...\n" + SEPARATOR)


    test_data = np.load('saved_data/test' + str(imgSize) + '.npy')
    test_id = np.load('saved_data/test_id.npy')


    print("\nPredicting with model...\n" + SEPARATOR)
    pred = model.predict_proba(test_data)

    df = pd.DataFrame(pred, columns=['Type_1', 'Type_2', 'Type_3'])
    df['image_name'] = test_id

    df.to_csv("../submission/Test001_Marek_" + timeStamp + ".csv", index=False)


if __name__ == '__main__':
    main()
