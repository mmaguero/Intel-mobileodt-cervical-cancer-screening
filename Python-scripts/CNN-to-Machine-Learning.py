# https://github.com/fchollet/keras/issues/431#issuecomment-213702147

import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from time import sleep

from extract_features import create_feature_extractor

imgSize = 64
loadFeatures = True
RDM = 17
useAditional = True
dataAugmentation = True

clasif = "SVM"  # RF/SVM
SEPARATOR = "=============================================================" + \
            "==================="


def svc(traindata, trainlabel, validData, validLabel, testData):
    print("Start training SVM...\n" + SEPARATOR)
    svcClf = SVC(kernel="rbf", verbose=True, decision_function_shape='ovo', probability=True, cache_size=1500)
    svcClf.fit(traindata, trainlabel)

    score = svcClf.score(validData, validLabel)
    sleep(1)
    print("Mean validation accuracy: " + str(score))
    pred_test = svcClf.predict_proba(testData)

    return pred_test


def rf(traindata, trainlabel, validData, validLabel, testData):
    print("Start training Random Forest...\n" + SEPARATOR)
    rfClf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=RDM, verbose=20,
                                   criterion='gini')
    rfClf.fit(traindata, trainlabel)

    score = rfClf.score(validData, validLabel)
    sleep(1)
    print("Mean validation accuracy: " + str(score))

    pred_test = rfClf.predict_proba(testData)

    return pred_test


if __name__ == '__main__':

    if (loadFeatures is False):
        create_feature_extractor()

    print("Loading features...\n" + SEPARATOR)
    if (dataAugmentation):
        if (useAditional):

            feature_train = np.load('saved_data/feaExt_DATrain' + str(imgSize) + '.npy')
            feature_valid = np.load('saved_data/feaExt_DAValid' + str(imgSize) + '.npy')
            train_target = np.load('saved_data/feaExt_DATrain' + str(imgSize) + '_target.npy')
            valid_target = np.load('saved_data/feaExt_DAValid' + str(imgSize) + '_target.npy')
            feature_test = np.load('saved_data/feaExt_test' + str(imgSize) + '.npy')
            test_id = np.load('saved_data/test_id.npy')
        else:
            feature_train = np.load('saved_data/fea_DATrain' + str(imgSize) + '.npy')
            feature_valid = np.load('saved_data/fea_DAValid' + str(imgSize) + '.npy')
            train_target = np.load('saved_data/fea_DATrain' + str(imgSize) + '_target.npy')
            valid_target = np.load('saved_data/fea_DAValid' + str(imgSize) + '_target.npy')
            feature_test = np.load('saved_data/fea_test' + str(imgSize) + '.npy')
            test_id = np.load('saved_data/test_id.npy')

    else:
        if (useAditional):
            feature_train = np.load('saved_data/feaExt_Train' + str(imgSize) + '.npy')
            feature_valid = np.load('saved_data/feaExt_Valid' + str(imgSize) + '.npy')
            train_target = np.load('saved_data/feaExt_Train' + str(imgSize) + '_target.npy')
            valid_target = np.load('saved_data/feaExt_Valid' + str(imgSize) + '_target.npy')
            feature_test = np.load('saved_data/feaExt_test' + str(imgSize) + '.npy')
            test_id = np.load('saved_data/test_id.npy')
        else:
            feature_train = np.load('saved_data/fea_Train' + str(imgSize) + '.npy')
            feature_valid = np.load('saved_data/fea_Valid' + str(imgSize) + '.npy')
            train_target = np.load('saved_data/fea_Train' + str(imgSize) + '_target.npy')
            valid_target = np.load('saved_data/fea_Valid' + str(imgSize) + '_target.npy')
            feature_test = np.load('saved_data/fea_test' + str(imgSize) + '.npy')
            test_id = np.load('saved_data/test_id.npy')


    # Turn feature maps into 2dim arrays

    print("Train features shape: "+str(feature_train.shape))
    print("Train labels shape: "+str(train_target.shape))
    if (dataAugmentation):
        nsamples, nx, ny, nz, nn = feature_train.shape
        d2_train_features = feature_train.reshape((nsamples * nx, ny * nz * nn))
        nsamples, nx = train_target.shape
        d1_train_target = train_target.reshape((nsamples * nx))
    else:
        nsamples, nx, ny, nz = feature_train.shape
        d2_train_features = feature_train.reshape((nsamples, nx * ny * nz ))
        d1_train_target = train_target
    print("Train features 2D shape: "+str(d2_train_features.shape))
    print("Train labels 1D shape: "+str(d1_train_target.shape))


    print("Valid features shape: "+str(feature_valid.shape))
    print("Valid labels shape: "+str(valid_target.shape))
    if (dataAugmentation):
        nsamples, nx, ny, nz, nn = feature_valid.shape
        d2_valid_features = feature_valid.reshape((nsamples * nx, ny * nz * nn))
        nsamples, nx = valid_target.shape
        d1_valid_target = valid_target.reshape((nsamples * nx))
    else:
        nsamples, nx, ny, nz = feature_valid.shape
        d2_valid_features = feature_valid.reshape((nsamples, nx * ny * nz))
        d1_valid_target = valid_target
    print("Valid features 2D shape: "+str(d2_valid_features.shape))
    print("Valid labels 1D shape: "+str(d1_valid_target.shape))


    print("Test features shape: "+str(feature_test.shape))
    nsamples, nx, ny, nz = feature_test.shape
    d2_test_features = feature_test.reshape((nsamples, nx * ny * nz))
    print("Test features 2D shape: "+str(d2_test_features.shape))

    print("Normalizing features...\n" + SEPARATOR)

    # Normalize features
    scaler = MinMaxScaler()
    d2_train_features = scaler.fit_transform(d2_train_features)
    d2_valid_features = scaler.fit_transform(d2_valid_features)
    d2_test_features = scaler.fit_transform(d2_test_features)

    if (clasif == "SVM"):
        pred = svc(d2_train_features, d1_train_target, d2_valid_features, d1_valid_target, d2_test_features)
    elif (clasif == "RF"):
        pred = rf(d2_train_features, d1_train_target, d2_valid_features, d1_valid_target, d2_test_features)
    else:
        print("Error, clasificador no instanciado")
        exit(0)
    print("\nPredicting with model...\n" + SEPARATOR)
    currentDate = datetime.today()
    timeStamp = currentDate.strftime("%d-%m-%Y_%H-%M")
    df = pd.DataFrame(pred, columns=['Type_1', 'Type_2', 'Type_3'])
    df['image_name'] = test_id

    if dataAugmentation:
        if useAditional:
            df.to_csv("../submission/" + clasif + "_ExtDA_" + timeStamp + ".csv", index=False)
        else:
            df.to_csv("../submission/" + clasif + "_DA_" + timeStamp + ".csv", index=False)
    else:
        if useAditional:
            df.to_csv("../submission/" + clasif + "_Ext_" + timeStamp + ".csv", index=False)
        else:
            df.to_csv("../submission/" + clasif + "_" + timeStamp + ".csv", index=False)
