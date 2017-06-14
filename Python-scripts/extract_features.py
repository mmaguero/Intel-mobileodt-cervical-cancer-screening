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
batchSize = 32
percentTrainForValidation = 0.1
useCustomPretrainedModels = True
RDM = 17
dataAugmentation = False
loadPreviousModel = True
pathToPreviousModel = ""
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

    print("\nLoading test data...\n" + SEPARATOR)
    if (keepAspectRatio):
        test_data = np.load('saved_data/test' + str(imgSize) + '_OrigAspectRatio.npy')
        test_id = np.load('saved_data/test_id.npy')
    else:
        test_data = np.load('saved_data/test' + str(imgSize) + '.npy')
        test_id = np.load('saved_data/test_id.npy')


    if(dataAugmentation):
        print("\nMaking data augmentation...\n" + SEPARATOR)
        datagen = da.prepareDataAugmentation(train_data=train_data)

    print("\nCreating model...\n" + SEPARATOR)
    if (loadPreviousModel):
        model = load_model(pathToPreviousModel)
        print("Loaded model from: " + pathToPreviousModel)
    else:
        model = create_pretrained_model(ftModel, ftApply)
    # loading VGG16 model weights
    model = VGG16(weights='imagenet', include_top=False, input_shape=(3, imgSize, imgSize))

    # Extracting features from the train dataset using the VGG16 pre-trained model
    print("\nGenerating features...\n" + SEPARATOR)
    if(dataAugmentation):
        #predict_generator(self, generator, steps, max_q_size=10, workers=1, pickle_safe=False, verbose=1)
        # TODO dar mas imagenes
        features_train=model.predict_generator(datagen.flow(x_train, y_train, batch_size=batchSize, shuffle=True), len(x_train), verbose=1)
        features_valid = model.predict_generator(datagen.flow(x_val_train, y_val_train, batch_size=batchSize, shuffle=True),
                                                 len(x_val_train), verbose=1)
    else:
        features_train=model.predict(x_train, batch_size=batchSize, verbose=1)
        features_valid=model.predict(x_val_train, batch_size=batchSize, verbose=1 )


    if(dataAugmentation):
        np.save('saved_data/feaExt_DATrain' + str(imgSize) + '.npy', features_train,
                    allow_pickle=True, fix_imports=True)
        np.save('saved_data/feaExt_DAValid' + str(imgSize) + '.npy', features_valid,
                    allow_pickle=True, fix_imports=True)
    else:
        np.save('saved_data/feaExt_Train' + str(imgSize) + '.npy', features_train,
                    allow_pickle=True, fix_imports=True)
        np.save('saved_data/feaExt_Train' + str(imgSize) + '_target.npy', y_train,
                allow_pickle=True, fix_imports=True)
        np.save('saved_data/feaExt_Valid' + str(imgSize) + '.npy', features_valid,
                    allow_pickle=True, fix_imports=True)
        np.save('saved_data/feaExt_Valid' + str(imgSize) + '_target.npy', y_val_train,
                allow_pickle=True, fix_imports=True)

    return features_train
    
if __name__ == '__main__':
    feature = create_feature_extractor()
