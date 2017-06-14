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
from sklearn.model_selection import GridSearchCV
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold

from image_utils import ImageUtils
from data_augmentation import DataAugmentation as da

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list
# the files in the input directory

# To Avoid Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

imgSize = 64
prepareData = False
useAditional = True
keepAspectRatio = True
useKaggleData = False
saveNetArchImage = False
NumEpoch = 30
batchSize = 32
percentTrainForValidation = 0.05
loadPreviousModel = False
pathToPreviousModel = "saved_data/scratch_model_ep05_10-06-2017_22-08.hdf5"
onlyEvaluate = False
hiperParamOpt = True
seed = 17

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
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=opt_,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def evaluateModel(model, testData, testLabels):
    score = model.evaluate(testData, testLabels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def hiperParametersOptimization(model, train, labels):
    '''
    Reference: http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
    :return:
    '''
    optimizers = ['adadelta', 'adamax', 'adam']
    batch_size = np.array([16, 32, 64])
    epochs = np.array([NumEpoch])
    param_grid = dict(batch_size=batch_size, epochs=epochs, opt_=optimizers)
    # evaluate using 10-fold cross validation
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    currentDate = datetime.today()
    timeStamp = currentDate.strftime("%d-%m-%Y_%H-%M")
    checkPoint = ModelCheckpoint(
        "saved_data/scratch_model_loss{val_loss:.4f}_ep{epoch:02d}_" + timeStamp + ".hdf5",
        save_best_only=True)
    callbackList = [checkPoint]
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_log_loss', cv=kfold, verbose=20,
                        fit_params={'callbacks': callbackList})
    grid_result = grid.fit(train, labels, )
    return grid_result


def main():
    if (prepareData):
        imgUtils = ImageUtils(imgSize, useAditional=useAditional, keepAspectRatio=keepAspectRatio,
                              useKaggleData=useKaggleData)
        imgUtils.dataPreparation()

    K.set_image_data_format('channels_first')
    K.set_floatx('float32')

    np.random.seed(seed)

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

    print("\nTraining Set shape (num Instances, RGB chanels, width, height): " + str(
        x_train.shape) + "\nTraining labels: " + str(y_train.shape) + "\nValidating set shape: " + str(
        x_val_train.shape) + "\nValidating set labels: " + str(
        y_val_train.shape) + "\n" + SEPARATOR)

    print("\nMaking data augmentation...\n" + SEPARATOR)
    datagen = da.prepareDataAugmentation(train_data=train_data)

    currentDate = datetime.today()
    timeStamp = currentDate.strftime("%d-%m-%Y_%H-%M")
    print("\nCreating model...\n" + SEPARATOR)
    if (loadPreviousModel):
        model = load_model(pathToPreviousModel)
        print("Loaded model from: " + pathToPreviousModel)
        model.summary()
    else:
        if (hiperParamOpt):
            print("\nHyperparameter optimization...\n" + SEPARATOR)
            model = KerasClassifier(build_fn=create_model, epochs=NumEpoch, batch_size=batchSize,
                                    validation_split=percentTrainForValidation)
            grid_result = hiperParametersOptimization(model, x_train, y_train)
            # summarize results
            print("Best score: %f using parameters %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
            grid_result.best_estimator_.model.save("saved_data/GridCV_Best_estimator"+timeStamp+".h5")
            model = grid_result

        else:
            model = create_model()

    if (saveNetArchImage):
        if (hiperParamOpt):
            plot_model(grid_result.best_estimator, to_file='saved_data/model_' + timeStamp + '.png')
        else:
            plot_model(model, to_file='saved_data/model_' + timeStamp + '.png')

    if (onlyEvaluate):

        print("\nEvaluating Model...\n" + SEPARATOR)
        evaluateModel(model, x_val_train, y_val_train)

    else:
        if hiperParamOpt is False:
            fitKerasModel(datagen, model, timeStamp, x_train, x_val_train, y_train, y_val_train)

    makePrediction(model, timeStamp)


def fitKerasModel(datagen, model, timeStamp, x_train, x_val_train, y_train, y_val_train):
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
                        callbacks=[checkPoint])  # , verbose=2)


def makePrediction(model, timeStamp):
    print("\nLoading test data...\n" + SEPARATOR)
    if (keepAspectRatio):
        test_data = np.load('saved_data/test' + str(imgSize) + '_OrigAspectRatio.npy')
        test_id = np.load('saved_data/test_id.npy')
    else:
        test_data = np.load('saved_data/test' + str(imgSize) + '.npy')
        test_id = np.load('saved_data/test_id.npy')
    print("\nPredicting with model...\n" + SEPARATOR)
    pred = model.predict_proba(test_data)
    df = pd.DataFrame(pred, columns=['Type_1', 'Type_2', 'Type_3'])
    df['image_name'] = test_id
    df.to_csv("../submission/Test001_Marek_" + timeStamp + ".csv", index=False)


if __name__ == '__main__':
    main()
