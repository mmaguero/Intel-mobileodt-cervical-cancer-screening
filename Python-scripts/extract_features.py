# https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/

# importing required libraries
import numpy as np

from keras import backend as K
from sklearn.model_selection import train_test_split

from image_utils import ImageUtils
from data_augmentation import DataAugmentation as da

imgSize = 64
prepareData = False
useAditional = True
keepAspectRatio = True
useKaggleData = False
batchSize = 128
percentTrainForValidation = 0.3
useCustomPretrainedModels = True
RDM = 17
dataAugmentation = False

SEPARATOR = "=============================================================" + \
            "==================="


if(useCustomPretrainedModels):
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

    if(dataAugmentation):
        print("\nMaking data augmentation...\n" + SEPARATOR)
        datagen = da.prepareDataAugmentation(train_data=train_data)

    # loading VGG16 model weights
    model = VGG16(weights='imagenet', include_top=False, input_shape=(3, imgSize, imgSize))

    # Extracting features from the train dataset using the VGG16 pre-trained model
    print("\nPredicting with model...\n" + SEPARATOR)
    if(dataAugmentation):
        #predict_generator(self, generator, steps, max_q_size=10, workers=1, pickle_safe=False, verbose=1)
        features_train=model.predict_generator(datagen.flow(x_train, y_train, batch_size=batchSize, shuffle=True), len(x_train), verbose=1)
    else:
        features_train=model.predict(train_data, batch_size=batchSize, verbose=1)

    train_x=features_train#.reshape(49000,25088)

    # converting target variable to array

    train_y=train_target#np.asarray(train['label'])

    # creating training and validation set
    X_train, X_valid, Y_train, Y_valid=train_test_split(train_x,train_y,test_size=percentTrainForValidation, random_state=RDM)

    #return X_train, X_valid, Y_train, Y_valid
    if(dataAugmentation):
        np.save('saved_data/feaExt_DATrain' + str(imgSize) + '.npy', X_train,
                    allow_pickle=True, fix_imports=True)
        np.save('saved_data/feaExt_DAValid' + str(imgSize) + '.npy', X_valid,
                    allow_pickle=True, fix_imports=True)
    else:
        np.save('saved_data/feaExt_Train' + str(imgSize) + '.npy', X_train,
                    allow_pickle=True, fix_imports=True)
        np.save('saved_data/feaExt_Valid' + str(imgSize) + '.npy', X_valid,
                    allow_pickle=True, fix_imports=True) 

    return features_train
    
if __name__ == '__main__':
    feature = create_feature_extractor()
