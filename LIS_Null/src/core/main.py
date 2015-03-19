'''
Created on Mar 7, 2015

@author: jonathan
'''

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
#     A = np.concatenate([complex_vec(x) for x in vec])
#     A = np.concatenate([[1],A])
#     A = np.atleast_2d(A)
#     A = np.dot(A.T, A)
#     return np.concatenate(A)

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

def read_data(inpath):
    X = []
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
#             t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            X.append(get_features(row[0:9],row[9:]))

    return np.atleast_2d(X)


X = read_data('train.csv')
Y = np.genfromtxt('train_y.csv', delimiter=',')
# select_features(X, Y)
print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)
print X[0]

Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.75)
print('Shape of Xtrain:', Xtrain.shape)
print('Shape of Ytrain:', Ytrain.shape)
print('Shape of Xtest:', Xtest.shape)
print('Shape of Ytest:', Ytest.shape)

Xval = read_data('validate.csv')
# Xval2 = read_data('test.csv')

# regressor = sken.GradientBoostingRegressor()
# regressor_svc = svm.SVC()
regressor = sken.RandomForestClassifier()
regressor.n_estimators = 256
regressor.min_samples_leaf = 1
regressor.min_samples_split = 2
regressor.n_classes_ = [7,3]
regressor.fit(X,Y)
Ypred = regressor.predict(Xval)
print('predicted result validate.csv')
np.savetxt('result_validate_quick.txt', Ypred, delimiter=',', fmt='%i')
# Ypred2 = regressor.predict(Xval2)
# print('predicted result of test.csv')
# np.savetxt('result_test_quick.txt', Ypred2)
regressor.fit(Xtrain, Ytrain)
# print('coefficients =', regressor.coef_)
# print('intercept =', regressor.intercept_)

# plt.plot(Xtrain[:, 0], Ytrain, 'bo')
# plt.xlim([-0.5, 23.5])
# plt.ylim([0, 1000])
# plt.show()

# Hplot = range(25)
# Xplot = np.atleast_2d([get_features(x) for x in Hplot])
# Yplot = regressor.predict(Xplot)
# plt.plot(Xtrain[:, 0], Ytrain, 'bo')
# plt.plot(Hplot, Yplot, 'r', linewidth=3)
# plt.xlim([-0.5, 23.5])
# plt.ylim([0, 1000])
# plt.show()

Ypred = regressor.predict(Xtest)
print('score of random forest=', multilabelscore(Ytest, Ypred))

# reg = svm.SVR()
# reg.fit(X,Y)
# regY = reg.predict(Xval)
# np.savetxt('validate_SVR.csv', regY)

scorefun = skmet.make_scorer(multilabelscore)
# scores = skcv.cross_val_score(regressor, X, Y, scoring=scorefun, cv=5)
# print('C-V score =', np.mean(scores), '+/-', np.std(scores))

# regressor_svr = svm.SVR()
regressor_cv = sken.RandomForestClassifier()
regressor_cv.n_estimators = 64
regressor_cv.n_classes_ = [7,3]
param_grid = {'min_samples_split': np.arange(2,10)}
neg_scorefun = skmet.make_scorer(lambda x, y: -multilabelscore(x, y))
grid_search = skgs.GridSearchCV(regressor_cv, param_grid, scoring=neg_scorefun, cv=5)
grid_search.fit(Xtrain, Ytrain)

best = grid_search.best_estimator_
print(best)
print('best score =', -grid_search.best_score_)

#Print result to file
Ypred = best.predict(Xval)
np.savetxt('result_validate.txt', Ypred)