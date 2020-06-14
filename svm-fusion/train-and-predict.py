"""
SVM Fusion of spatial and temporal streams 


"""
import numpy as np
import os, glob
import pandas as pd
from random import shuffle
from sklearn.preprocessing import StandardScaler
from PIL import Image
import pandas as pd
from pathlib import Path
import sys
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score
from sklearn.svm import SVC

def generate_dataset(mode, clf_key):
    ps_path = '/mnt/data3/crop-classification/4_ml/purely_spatial/{}-<Section: crop-{}>-probs.csv'.format(mode, clf_key)
    pt_path = '/mnt/data3/crop-classification/4_ml/purely-temporal/pt-{}-<Section: temporal>-probs.csv'.format(mode)    
    ps_df = pd.read_csv(ps_path)
    pt_df = pd.read_csv(pt_path)
    all_df = pd.merge(ps_df, pt_df, on='fname', how='inner')
    print(all_df.columns)
    all_df_X = all_df[['SP0', 'SP1', 'SP2', 'SP3', 'SP4', 'SP5', 'TP0', 'TP1', 'TP2', 'TP3', 'TP4', 'TP5']].values
    all_df_Y = all_df['Class_x'].values
    return [all_df_X, all_df_Y]


def train_predict(clf_key, classifier, param_grid, trainX, trainY, valX, valY, testX, testY):
    all_tr_val_X = np.vstack((trainX, valX))
    all_tr_val_Y = np.hstack((trainY, valY))
    print(all_tr_val_X.shape, all_tr_val_Y.shape)
    fold_meta = np.zeros(all_tr_val_X.shape[0])
    fold_meta[0:trainX.shape[0]] = -1
    cv = PredefinedSplit(test_fold=fold_meta)
    gcv = GridSearchCV(estimator = classifier, param_grid = param_grid, cv=cv, verbose=0, n_jobs=2, scoring='accuracy')
    gcv.fit(all_tr_val_X, all_tr_val_Y)
    predictions = gcv.predict(testX)
    cm = confusion_matrix(testY, predictions)
    classes_lst = ['Corn', 'Cotton', 'Soy', 'Spring Wheat', 'Winter Wheat', 'Barley']
    creport = classification_report(y_true = testY, y_pred=predictions, target_names = classes_lst, digits = 4, output_dict = True)
    creport_df = pd.DataFrame(creport).transpose()
    acc = accuracy_score(testY, predictions)
    print(creport)
    kappa_score = cohen_kappa_score(testY, predictions)
    print('Classifier : {}'.format(clf_key))
    print('best params: {}'.format(gcv.best_params_))
    print('Accuracy is {}\n Kappa Score is {}\n confusion matrix is {}\n clf report is {}'.format(acc, kappa_score, cm, creport))



if __name__=="__main__":
    classifiers_grid = {
                        'svc':{
                            'C': [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000],
                            'gamma':[0.1, 1, 2, 10, 'auto']
                        }

                        }

    for clf_key in ['vgg16', 'resnet', 'densenet']:
        trainX, trainY = generate_dataset('train', clf_key)
        valX, valY = generate_dataset('val', clf_key)
        testX, testY = generate_dataset('test', clf_key)
        print('Train size = {}, val size = {}, Test size = {}'.format(trainX.shape[0], valX.shape[0], testX.shape[0]))
        train_predict(clf_key, SVC(), classifiers_grid['svc'], trainX, trainY, valX, valY, testX, testY)
