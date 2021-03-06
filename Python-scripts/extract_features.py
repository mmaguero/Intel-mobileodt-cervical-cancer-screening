# https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/

# importing required libraries
import numpy as np

from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.applications.inception_v3 import InceptionV3
from keras.models import load_model, Model
from image_utils import ImageUtils
from data_augmentation import DataAugmentation as da

imgSize = 64
prepareData = False
useAditional = True
keepAspectRatio = False
useKaggleData = False
batchSize = 12
percentTrainForValidation = 0.015
useCustomPretrainedModels = True
RDM = 17
dataAugmentation = True
loadPreviousModel = True
pathToPreviousModel = "saved_data/VGG16_fine-tunned_ep00_13-06-2017_17-26.hdf5"
ftModel = "VGG16"  # IV3/VGG16/ = InceptionV3[Min.139|Def.299]/VGG16[Min.48|Def.224]
SEPARATOR = "=============================================================" + \
            "==================="

if (useCustomPretrainedModels):
    from pretrained import VGG16
else:
    from keras.applications.vgg16 import VGG16


def create_feature_extractor():
    if (prepareData):
        imgUtils = ImageUtils(imgSize, useAditional=useAditional, keepAspectRatio=keepAspectRatio,
                              useKaggleData=useKaggleData)
        imgUtils.dataPreparation()()

    K.set_image_data_format('channels_first')
    K.set_floatx('float32')

    np.random.seed(RDM)

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
        random_state=RDM)

    print("\nLoading test data...\n" + SEPARATOR)
    if (keepAspectRatio):
        test_data = np.load('saved_data/test' + str(imgSize) + '_OrigAspectRatio.npy')
    else:
        test_data = np.load('saved_data/test' + str(imgSize) + '.npy')

    if (dataAugmentation):
        print("\nMaking data augmentation...\n" + SEPARATOR)
        datagen = da.prepareDataAugmentation(train_data=train_data)

    print("\nCreating model...\n" + SEPARATOR)
    if (loadPreviousModel):
        baseModel = load_model(pathToPreviousModel)
        print("Loaded model from: " + pathToPreviousModel)

        if (ftModel == "VGG16"):
            model = Model(input=baseModel.input, outputs=baseModel.get_layer("block5_pool").output)
        elif (ftModel == "IV3"):
            model = Model(input=baseModel.input, outputs=baseModel.get_layer("mixed10").output)
    else:
        if (ftModel == "VGG16"):
            # loading VGG16 model weights
            model = VGG16(weights='imagenet', include_top=False, input_shape=(3, imgSize, imgSize))
        elif (ftModel == "IV3"):
            model = InceptionV3(weights='imagenet', include_top=False, input_shape=(3, imgSize, imgSize))

    # Extracting features from the train dataset using the VGG16 pre-trained model
    print("\nGenerating features...\n" + SEPARATOR)
    print("\nTraining features...\n")
    if (dataAugmentation):
        # predict_generator(self, generator, steps, max_q_size=10, workers=1, pickle_safe=False, verbose=1)
        # TODO dar mas imagenes7

        batches = 0
        features_train = []
        train_labels = []
        for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=batchSize, shuffle=True):
            features_batch = model.predict_on_batch(x_batch)
            features_train.append(features_batch)
            train_labels.append(y_batch)
            batches += 1
            print("Batches: " + str(batches) + '/' + str(len(x_train)))
            if batches >= len(x_train):
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        print("\nValidation features...\n")

        batches = 0
        features_valid = []
        valid_labels = []
        for x_batch, y_batch in datagen.flow(x_val_train, y_val_train, batch_size=batchSize, shuffle=True):
            features_batch = model.predict_on_batch(x_batch)
            features_valid.append(features_batch)
            valid_labels.append(y_batch)
            batches += 1
            print("Batches: " + str(batches) + '/' + str(len(x_val_train)))
            if batches >= len(x_val_train) // batchSize:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        print("\nTest features...\n")

        features_test = model.predict(test_data, batch_size=batchSize, verbose=1)

    else:
        features_train = model.predict(x_train, batch_size=batchSize, verbose=1)
        print("\nValidation features...\n")
        features_valid = model.predict(x_val_train, batch_size=batchSize, verbose=1)
        print("\nTest features...\n")
        features_test = model.predict(test_data, batch_size=batchSize, verbose=1)

    if (dataAugmentation):
        if (useAditional):
            if (keepAspectRatio):
                np.save('saved_data/feaExt_DATrain' + str(imgSize) + '.npy', features_train,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/feaExt_DATrain' + str(imgSize) + '_target.npy', train_labels,
                        allow_pickle=True, fix_imports=True)

                np.save('saved_data/feaExt_DAValid' + str(imgSize) + '.npy', features_valid,
                        allow_pickle=True, fix_imports=True)

                np.save('saved_data/feaExt_DAValid' + str(imgSize) + '_target.npy', valid_labels,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/feaExt_test' + str(imgSize) + '.npy', features_test,
                        allow_pickle=True, fix_imports=True)
            else:
                np.save('saved_data/feaExt_DA_NAR_Train' + str(imgSize) + '.npy', features_train,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/feaExt_DA_NAR_Train' + str(imgSize) + '_target.npy', train_labels,
                        allow_pickle=True, fix_imports=True)

                np.save('saved_data/feaExt_DA_NAR_Valid' + str(imgSize) + '.npy', features_valid,
                        allow_pickle=True, fix_imports=True)

                np.save('saved_data/feaExt_DA_NAR_Valid' + str(imgSize) + '_target.npy', valid_labels,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/feaExt__NAR_test' + str(imgSize) + '.npy', features_test,
                        allow_pickle=True, fix_imports=True)
        else:
            if (keepAspectRatio):
                np.save('saved_data/fea_DATrain' + str(imgSize) + '.npy', features_train,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/fea_DATrain' + str(imgSize) + '_target.npy', train_labels,
                        allow_pickle=True, fix_imports=True)

                np.save('saved_data/fea_DAValid' + str(imgSize) + '.npy', features_valid,
                        allow_pickle=True, fix_imports=True)

                np.save('saved_data/fea_DAValid' + str(imgSize) + '_target.npy', valid_labels,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/fea_test' + str(imgSize) + '.npy', features_test,
                        allow_pickle=True, fix_imports=True)
            else:
                np.save('saved_data/fea_DA_NAR_Train' + str(imgSize) + '.npy', features_train,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/fea_DA_NAR_Train' + str(imgSize) + '_target.npy', train_labels,
                        allow_pickle=True, fix_imports=True)

                np.save('saved_data/fea_DA_NAR_Valid' + str(imgSize) + '.npy', features_valid,
                        allow_pickle=True, fix_imports=True)

                np.save('saved_data/fea_DA_NAR_Valid' + str(imgSize) + '_target.npy', valid_labels,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/fea__NAR_test' + str(imgSize) + '.npy', features_test,
                        allow_pickle=True, fix_imports=True)
    else:
        if (useAditional):
            if (keepAspectRatio):
                np.save('saved_data/feaExt_Train' + str(imgSize) + '.npy', features_train,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/feaExt_Train' + str(imgSize) + '_target.npy', y_train,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/feaExt_Valid' + str(imgSize) + '.npy', features_valid,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/feaExt_Valid' + str(imgSize) + '_target.npy', y_val_train,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/feaExt_test' + str(imgSize) + '.npy', features_test,
                        allow_pickle=True, fix_imports=True)
            else:
                np.save('saved_data/feaExt_NAR_Train' + str(imgSize) + '.npy', features_train,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/feaExt_NAR_Train' + str(imgSize) + '_target.npy', y_train,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/feaExt_NAR_Valid' + str(imgSize) + '.npy', features_valid,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/feaExt_NAR_Valid' + str(imgSize) + '_target.npy', y_val_train,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/feaExt_NAR_test' + str(imgSize) + '.npy', features_test,
                        allow_pickle=True, fix_imports=True)
        else:
            if (keepAspectRatio):
                np.save('saved_data/fea_Train' + str(imgSize) + '.npy', features_train,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/fea_Train' + str(imgSize) + '_target.npy', y_train,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/fea_Valid' + str(imgSize) + '.npy', features_valid,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/fea_Valid' + str(imgSize) + '_target.npy', y_val_train,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/fea_test' + str(imgSize) + '.npy', features_test,
                        allow_pickle=True, fix_imports=True)
            else:
                np.save('saved_data/fea_NAR_Train' + str(imgSize) + '.npy', features_train,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/fea_NAR_Train' + str(imgSize) + '_target.npy', y_train,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/fea_NAR_Valid' + str(imgSize) + '.npy', features_valid,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/fea_NAR_Valid' + str(imgSize) + '_target.npy', y_val_train,
                        allow_pickle=True, fix_imports=True)
                np.save('saved_data/fea_NAR_test' + str(imgSize) + '.npy', features_test,
                        allow_pickle=True, fix_imports=True)


if __name__ == '__main__':
    feature = create_feature_extractor()
