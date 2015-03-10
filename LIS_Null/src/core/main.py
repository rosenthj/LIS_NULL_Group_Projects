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

MonthsTable = [0,3,3,6,1,4,6,2,5,0,3,5]

def get_weekday(y, m, d):
    return np.mod(y-2000+m+d+((y-2000)/4) + 6, 7)

def f_log(x):
    return np.log2(np.abs(x)+0.01)

def create_vector(a):
    return [a, a*a, a*a*a, f_log(a), a*f_log(a), np.exp2(a), np.sin(a), a*np.sin(a)]

def get_cat_vec(c, num_cat):
    A = [0] * num_cat
    A[c] = 1
    return A

def get_features(h, r, w):
    return np.concatenate([[1, h, h*h, h*h*h],[r/5]])
#     A = np.concatenate([[h, h*h, h*h*h], r])
#     return A # np.concatenate([create_vector(a) for a in A[:]])

def logscore(gtruth, pred):
    pred = np.clip(pred, 0, np.inf)
    logdif = np.log(1 + gtruth) - np.log(1 + pred)
    return np.sqrt(np.mean(np.square(logdif)))
 

def read_data(inpath):
    X = []
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
#             X.append(get_features(t.hour, [float(x) for x in row[1:]]))
            X.append(get_features(t.hour, t.weekday(), int(row[2])))

    return np.atleast_2d(X)

def select_features(X, y):
    svc = svm.SVR(kernel="linear")
    rfecv = skfs.RFECV(estimator=svc, step=1,
              scoring='accuracy')
    rfecv.fit(X, y)
#     print("Optimal number of features : %d" % rfecv.n_features_)


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

regressor = svm.SVR()
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
print('score of svr=', logscore(Ytest, Ypred))

# reg = svm.SVR()
# reg.fit(X,Y)
Xval = read_data('validate.csv')
# regY = reg.predict(Xval)
# np.savetxt('validate_SVR.csv', regY)

scorefun = skmet.make_scorer(logscore)
# scores = skcv.cross_val_score(regressor, X, Y, scoring=scorefun, cv=5)
# print('C-V score =', np.mean(scores), '+/-', np.std(scores))

regressor_svr = svm.SVR()
param_grid = {'epsilon': np.linspace(0, 1, 10)}
neg_scorefun = skmet.make_scorer(lambda x, y: -logscore(x, y))
grid_search = skgs.GridSearchCV(regressor_svr, param_grid, scoring=neg_scorefun, cv=5)
grid_search.fit(Xtrain, Ytrain)

best = grid_search.best_estimator_
print(best)
print('best score =', -grid_search.best_score_)

#Print result to file
Ypred = best.predict(Xval)
np.savetxt('result_validate.txt', Ypred)