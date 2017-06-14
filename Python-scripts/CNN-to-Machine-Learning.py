# https://github.com/fchollet/keras/issues/431#issuecomment-213702147

from __future__ import print_function
import theano
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from extract_features import create_feature_extractor

imgSize = 64
RDM = 17
loadFeatures = True

SEPARATOR = "=============================================================" + \
            "==================="

def svc(traindata,trainlabel,testdata):

    print("Start training SVM...\n" + SEPARATOR)
    svcClf = SVC(C=1.0,kernel="rbf")#,cache_size=900)
    svcClf.fit(traindata,trainlabel)

    pred_testlabel = svcClf.predict(testdata)

def rf(traindata,trainlabel,testdata):

    print("Start training Random Forest...\n" + SEPARATOR)
    rfClf = RandomForestClassifier(n_estimators=2,criterion='gini')
    rfClf.fit(traindata,trainlabel)

    pred_testlabel = rfClf.predict(testdata)

if __name__ == '__main__':

    if(loadFeatures):
        print("Load predict...\n" + SEPARATOR)
        feature = np.load('saved_data/feaExt_Train' + str(imgSize) + '.npy')
    else:
        feature = create_feature_extractor()

    layer = model.feature[-9]  # Adjust here to the right depth.
tensor = layer.get_output_at(0)
f = K.function(model.inputs + [K.learning_phase()], (tensor,))
X_train_svm = f(X_train) 

    print("train svm using FC-layer feature\n" + SEPARATOR)
    scaler = MinMaxScaler()
    feature = scaler.fit_transform(feature)

    svc(feature[0:900],feature[0:900],feature[900:])
    rf(feature[0:900],feature[0:900],feature[900:])

