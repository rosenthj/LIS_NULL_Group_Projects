'''
Created on Mar 7, 2015

@author: jonathan
'''
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing.data import MinMaxScaler

if __name__ == '__main__':
    pass

import sklearn.svm as svm
# Provides Matlab-style matrix operations
import numpy as np
# Provides Matlab-style plotting
import matplotlib.pylab as plt
# For reading and writing csv files
import csv
# For parsing date/time strings
import datetime
# For parsing .h5 file format
import h5py
# Contains linear models, e.g., linear regression, ridge regression, LASSO, etc.
import sklearn.linear_model as sklin
# Allows us to create custom scoring functions
import sklearn.metrics as skmet
# Provides train-test split, cross-validation, etc.
import sklearn.cross_validation as skcv
# Provides grid search functionality
import sklearn.grid_search as skgs
# Provides feature selection functionality
import sklearn.feature_selection as skfs
# Provides access to ensemble based classification and regression.
import sklearn.ensemble as sken
import sklearn.tree as sktree
import sklearn.neighbors as nn
import sklearn.preprocessing as skprep
import sklearn.neural_network as skneural
import sklearn.cluster as cluster
import sklearn.pipeline as skpipe
import sklearn.decomposition as skdec
import sklearn.naive_bayes as skbayes

MonthsTable = [0,3,3,6,1,4,6,2,5,0,3,5]

def get_weekday(y, m, d):
    return np.mod(y-2000+m+d+((y-2000)/4) + 6, 7)

def f_log(x):
    return np.log2(np.abs(x)+0.01)

def create_vector(a):
    return [a]#, f_log(a), np.exp2(a)]

def complex_vec(a):
    return [a]

def create_complex_vec(vec):
    return np.concatenate([complex_vec(x) for x in vec])

def get_cat_vec(c, num_cat, x=1):
    A = [0] * num_cat
    A[c] = x
    return A

def get_features(geomVec, catVec):
    return np.concatenate([create_complex_vec([int(x) for x in geomVec]), [int(x) for x in catVec]])

def logscore(gtruth, pred):
    pred = np.clip(pred, 0, np.inf)
    logdif = np.log(1 + gtruth) - np.log(1 + pred)
    return np.sqrt(np.mean(np.square(logdif)))

def elementsnotequal(x, y):
    if (x == y):
        return 0
    return 1

def labelsnotequal(x, y):
    return map(elementsnotequal,x, y)

def multilabelscore(gtruth, pred):
    return np.mean(map(labelsnotequal,gtruth, pred))

def singlelabelscore(gtruth, pred):
    return np.mean(map(elementsnotequal,gtruth, pred))

def read_data(inpath):
    X = []
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            X.append(get_features(row[0:9],row[9:]))

    return np.atleast_2d(X)

def read_h5data(inpath):
    return h5py.File(inpath, "r")["data"][...]

def read_h5labels(inpath):
    f = h5py.File(inpath, "r")
    return np.squeeze(np.asarray(f["label"]))

print('loading training data')
X = read_h5data('train.h5')
Y = read_h5labels('train.h5')

print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)

Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.5)
print 'Shape of Xtrain:', Xtrain.shape
print 'Shape of Ytrain:', Ytrain.shape
print 'Shape of Xtest:', Xtest.shape
print 'Shape of Ytest:', Ytest.shape

print 'loading validation data'
valX = read_h5data('validate.h5')
print 'Shape of valX:', valX.shape

# print 'loading test data'
# testX = read_h5data('test.h5')
# print 'Shape of testX:', testX.shape

# PREPROCESSING
# SCALING
minMaxScaler = skprep.MinMaxScaler()
# FEATURE SELECTION
varianceThresholdSelector = skfs.VarianceThreshold(threshold=(0))
percentileSelector = skfs.SelectPercentile(score_func=skfs.f_classif, percentile=20)
kBestSelector = skfs.SelectKBest(skfs.f_classif, 1000)
# FEATURE EXTRACTION
rbm = skneural.BernoulliRBM(random_state=0, learning_rate = 0.01, n_components = 100, n_iter = 5, verbose=True)
rbmPipe = skpipe.Pipeline(steps=[('scaling', minMaxScaler), ('rbm', rbm)])
nmf = skdec.NMF(n_components=150)
pca = skdec.PCA(n_components=450)
kernelPCA = skdec.KernelPCA(n_components=25)# Costs huge amounts of ram
randomizedPCA= skdec.RandomizedPCA()

# REGRESSORS
GradientBoostingRegressor = sken.GradientBoostingRegressor()
supportVectorRegressor = svm.SVR()

# CLASSIFIERS
supportVectorClassifier = svm.SVC()
linearSupportVectorClassifier = svm.LinearSVC()
nearestNeighborClassifier = nn.KNeighborsClassifier()
extraTreesClassifier = sken.ExtraTreesClassifier(n_estimators=16)
randomForestClassifier = sken.RandomForestClassifier(n_estimators=32)
logisticClassifier = sklin.LogisticRegression(C=80)
ridgeClassifier = sklin.RidgeClassifier(alpha=0.1, solver='svd')
bayes = skbayes.MultinomialNB()

# FEATURE UNION
featureUnion = skpipe.FeatureUnion(transformer_list=[('pca', pca)])

# PIPE DEFINITION
classifier = skpipe.Pipeline(steps=[('randomizedPCA', randomizedPCA),('estimator', randomForestClassifier)])
print 'Successfully prepared classifier pipeline!'

# GRID DEFINITION
# classifierSearcher = GridSearchCV(classifier, dict(pca__n_components=[100, 150, 200], estimator__alpha=[1,2,5]), verbose=2)
#    
# print 'fitting classifier pipeline grid on training data subset for accuracy estimate'
# classifierSearcher.fit(Xtrain, Ytrain)
# print 'best estimator:', classifierSearcher.best_estimator_
# classifier = classifierSearcher.best_estimator_
# print 'classifier pipe is predicting result of data'
# Ypred = classifier.predict(Xtest)
# print('score of classifier=', singlelabelscore(Ytest, Ypred))

print 'fitting classifier pipeline on training data'
classifier.fit(X, Y)
print 'classifier pipe is predicting result of data'
Ypred = classifier.predict(valX)
print('Ypred shape', Ypred.shape)
print('predicted result validate.csv')
np.savetxt('result_validate_quick.txt', Ypred, delimiter=',', fmt='%i')
