#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 00:27:49 2017

@author: badrawy
"""
import numpy as np
import itertools
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve
from matplotlib import pyplot as plt
from time import time
from sklearn.metrics import classification_report

def classify(clf,X_train,y_train,X_test,y_test):
    #train the model
    t0 = time()
    print("start fit")
    clf.fit(X_train,y_train)
    print ("Training time:", round(time()-t0, 3), "s")

    #test the model

    pred=clf.predict(X_test)
    print(accuracy_score(y_test, pred))

    print(classification_report(y_test, pred))
    
    
    
def plot_validation(train_scores, test_scores,param_range,cl_name,param_name):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    fig= plt.figure(figsize=(10.24,5.12),dpi=100)

    plt.title("Validation Curve with _"+cl_name)

    plt.xlabel("param _"+param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)



    

    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    
    plt.legend(loc="best")
    #plt.xticks(param_range)
    #plt.tick_params(labelbottom='off')    

    fig.savefig(cl_name+"_"+param_name+".png",dpi=fig.dpi)

    plt.show()
    
    
    
    
def data():
    X_train=np.genfromtxt('data/X_train.txt',dtype='float')
    X_test=np.genfromtxt('data/X_test.txt',dtype='float')
    y_train=np.genfromtxt('data/y_train.txt',dtype='float')
    y_test=np.genfromtxt('data/y_test.txt',dtype='float')
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return  X_train,y_train,X_test,y_test



def add_bias(X_train,X_test):
    ones=np.ones(shape=(X_train.shape[0],1))
    X_train=np.append(X_train, ones, axis=1)
    ones=np.ones(shape=(X_test.shape[0],1))
    X_test=np.append(X_test, ones, axis=1)
    return X_train,X_test


#find optimial hidden layer sizes
X_train,y_train,X_test,y_test=data()
X_train,X_test=add_bias(X_train,X_test)
mlp = MLPClassifier(random_state=101)
param_range_HL = [x for x in itertools.product((300,400,500,600,700),repeat=1)]+[x for x in itertools.product((300,400,500,600,700),repeat=2)]

train_scores, valid_scores =validation_curve(mlp,
                                             X_train,y_train,
                                            'hidden_layer_sizes',
                                             param_range_HL,cv=3,
                                             verbose=True,n_jobs=-1)

param_range_HL=np.arange(0,30)
plot_validation(train_scores,valid_scores,param_range_HL,'MLP','HLSizs')
classify(mlp,X_train,y_train,X_test,y_test)

#find optimial alpha
X_train,y_train,X_test,y_test=data()
X_train,X_test=add_bias(X_train,X_test)
mlp = MLPClassifier(random_state=101,hidden_layer_sizes=(700,))
param_range_alpha = [0.00001,0.001,0.1,1,10]

train_scores, valid_scores =validation_curve(mlp,
                                             X_train,y_train,
                                            'alpha',
                                             param_range_alpha,cv=3,
                                             verbose=True,n_jobs=-1)

plot_validation(train_scores,valid_scores,param_range_alpha,'MLP','alpha700')
classify(mlp,X_train,y_train,X_test,y_test)

#final optimal mlp
X_train,y_train,X_test,y_test=data()
X_train,X_test=add_bias(X_train,X_test)
mlp = MLPClassifier(random_state=101,hidden_layer_sizes=(700,),alpha=1)
classify(mlp,X_train,y_train,X_test,y_test)

#PCA for mLP
X_train,y_train,X_test,y_test=data()
pca= PCA(n_components=5)
pca=pca.fit(X_train)
X_train=pca.transform(X_train)
X_test=pca.transform(X_test)
X_train,X_test=add_bias(X_train,X_test)
mlp = MLPClassifier(random_state=101,hidden_layer_sizes=(700,),alpha=1)
classify(mlp,X_train,y_train,X_test,y_test)

X_train,y_train,X_test,y_test=data()
pca= PCA(n_components=50)
pca=pca.fit(X_train)
X_train=pca.transform(X_train)
X_test=pca.transform(X_test)
X_train,X_test=add_bias(X_train,X_test)
mlp = MLPClassifier(random_state=101,hidden_layer_sizes=(700,),alpha=1)
classify(mlp,X_train,y_train,X_test,y_test)

X_train,y_train,X_test,y_test=data()
pca= PCA(n_components=200)
pca=pca.fit(X_train)
X_train=pca.transform(X_train)
X_test=pca.transform(X_test)
X_train,X_test=add_bias(X_train,X_test)
mlp = MLPClassifier(random_state=101,hidden_layer_sizes=(700,),alpha=1)
classify(mlp,X_train,y_train,X_test,y_test)


X_train,y_train,X_test,y_test=data()
pca= PCA(n_components=500)
pca=pca.fit(X_train)
X_train=pca.transform(X_train)
X_test=pca.transform(X_test)
X_train,X_test=add_bias(X_train,X_test)
mlp = MLPClassifier(random_state=101,hidden_layer_sizes=(700,),alpha=1)
classify(mlp,X_train,y_train,X_test,y_test)



#############SVM##########################

#find optimal vlaue for gamma paramter for svm classifer
X_train,y_train,X_test,y_test=data()
svm=SVC(random_state=101)

param_range_gamma = [1,0.1,0.01,0.001,0.0001]
train_scores, valid_scores =validation_curve(svm,
                                             X_train,y_train,
                                             'gamma',
                                             param_range_gamma,cv=3,
                                             verbose=True,n_jobs=-1)

plot_validation(train_scores,valid_scores,param_range_gamma,"SVM","gamma")
classify(svm,X_train,y_train,X_test,y_test)


# find optimal value for C paramter for svm classifrer
#best gamma 10-3=0.001 bec validaiton scrore starts decreasing
X_train,y_train,X_test,y_test=data()
svm=SVC(random_state=101,gamma=0.001)

param_range_C = [0.1,1,10,100,1000]
train_scores, valid_scores =validation_curve(svm,
                                             X_train,y_train,
                                            'C',
                                             param_range_C,cv=3,
                                             verbose=True,n_jobs=-1)

plot_validation(train_scores,valid_scores,param_range_C,'SVM','C')
classify(svm,X_train,y_train,X_test,y_test)




#best c is 103 (1000) becuase train and valid is the same
#optimal svm clasisfar with best gamma and c
svm=SVC(random_state=101,gamma=0.001,C=1000)
X_train,y_train,X_test,y_test=data()
classify(svm,X_train,y_train,X_test,y_test)

#PCA for SVM
X_train,y_train,X_test,y_test=data()
pca= PCA(n_components=5)
pca=pca.fit(X_train)
X_train=pca.transform(X_train)
X_test=pca.transform(X_test)
svm=SVC(random_state=101,gamma=0.001,C=1000)
classify(svm,X_train,y_train,X_test,y_test)


X_train,y_train,X_test,y_test=data()
pca= PCA(n_components=50)
pca=pca.fit(X_train)
X_train=pca.transform(X_train)
X_test=pca.transform(X_test)
svm=SVC(random_state=101,gamma=0.001,C=1000)
classify(svm,X_train,y_train,X_test,y_test)


X_train,y_train,X_test,y_test=data()
pca= PCA(n_components=200)
pca=pca.fit(X_train)
X_train=pca.transform(X_train)
X_test=pca.transform(X_test)
svm=SVC(random_state=101,gamma=0.001,C=1000)
classify(svm,X_train,y_train,X_test,y_test)


X_train,y_train,X_test,y_test=data()
pca= PCA(n_components=500)
pca=pca.fit(X_train)
X_train=pca.transform(X_train)
X_test=pca.transform(X_test)
svm=SVC(random_state=101,gamma=0.001,C=1000)
classify(svm,X_train,y_train,X_test,y_test)














