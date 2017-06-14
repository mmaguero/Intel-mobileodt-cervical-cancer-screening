# https://github.com/fchollet/keras/issues/431#issuecomment-213702147

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from extract_features import create_feature_extractor

imgSize = 64
useAditional = True
keepAspectRatio = True
loadFeatures = True
RDM = 17
percentTrainForValidation = 0.15

clasif="RF" # RF/SVM
SEPARATOR = "=============================================================" + \
            "==================="


def svc(traindata, trainlabel, validData, validLabel):#, testdata):
    print("Start training SVM...\n" + SEPARATOR)
    svcClf = SVC(kernel="rbf", verbose=True, decision_function_shape='ovo', probability=True, cache_size=900)
    svcClf.fit(traindata, trainlabel)

    score=svcClf.score(validData, validLabel)
    print("Mean validation accuracy: "+str(score))
   # pred_testlabel = svcClf.predict(testdata)


def rf(traindata, trainlabel, validData, validLabel):
    print("Start training Random Forest...\n" + SEPARATOR)
    rfClf = RandomForestClassifier(n_estimators=1000, oob_score=True, n_jobs=4, random_state=RDM, verbose=20, criterion='gini')
    rfClf.fit(traindata, trainlabel)

    score =rfClf.score(validData, validLabel)
    print("Mean validation accuracy: "+str(score))

    #pred_testlabel = rfClf.predict(testdata)


if __name__ == '__main__':

    if (loadFeatures):
        print("Loading features...\n" + SEPARATOR)
        feature_train = np.load('saved_data/feaExt_Train' + str(imgSize) + '.npy')
        feature_valid = np.load('saved_data/feaExt_Valid' + str(imgSize) + '.npy')
        train_target = np.load('saved_data/feaExt_Train' + str(imgSize) + '_target.npy')
        valid_target = np.load('saved_data/feaExt_Valid' + str(imgSize) + '_target.npy')
        feature_test = np.load('saved_data/fea_Test' + str(imgSize) + '.npy')
        test_id = np.load('saved_data/test_id.npy')

    else:
        feature = create_feature_extractor()

   # print(feature)

    # Turn feature maps into 2dim arrays
    print(feature_train.shape)
    nsamples, nz, nx, ny = feature_train.shape
    print(nsamples, nz, nx, ny)
    d2_train_features = feature_train.reshape((nsamples, nx * ny * nz))

    nsamples, nz, nx, ny = feature_valid.shape
    d2_valid_features = feature_valid.reshape((nsamples, nx * ny * nz))

    #Normalize features
    scaler = MinMaxScaler()
    d2_train_features = scaler.fit_transform(d2_train_features)
    d2_valid_features = scaler.fit_transform(d2_valid_features)

    if(clasif == "SVM"):
        svc(d2_train_features, train_target, d2_valid_features, valid_target)
    elif(clasif == "RF"):
        rf (d2_train_features, train_target, d2_valid_features, valid_target)
    else:
        print("Error, clasificador no instanciado")
