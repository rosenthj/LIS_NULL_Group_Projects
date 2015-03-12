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

MonthsTable = [0,3,3,6,1,4,6,2,5,0,3,5]

def get_weekday(y, m, d):
    return np.mod(y-2000+m+d+((y-2000)/4) + 6, 7)

def f_log(x):
    return np.log2(np.abs(x)+0.01)

def create_vector(a):
    return [a]#, f_log(a), np.exp2(a)]

def get_cat_vec(c, num_cat, x=1):
    A = [0] * num_cat
    A[c] = x
    return A

def get_features(y, mon, h, m, r, w, wl, wr):
    return np.concatenate([[y, mon, h, h*h, h*h*h, h*h*h*h, m, r/5, r/4-r/6],get_cat_vec(r, 7) ,[wl, (r/5)*wl], get_cat_vec(w, 4),wr, [(r/5)*x for x in wr], get_cat_vec(w, 4, r/5)])
#     A = np.concatenate([[h, m, r/5], wl, wr])
#     A = np.concatenate([create_vector(a) for a in A[:]])
#     A = np.concatenate([[1],A])
#     A = np.atleast_2d(A)
#     A = np.dot(A.T, A)
#     A = np.concatenate(A)
#     return np.concatenate([[h*h*h,h*h*h*h],A, get_cat_vec(r, 7), get_cat_vec(w, 4)])

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
            X.append(get_features(t.year, t.month, t.hour*60+t.minute, t.minute, t.weekday(), int(row[2]), float(row[1]), [float(x) for x in row[3:]]))

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

Xval = read_data('validate.csv')

# regressor = sken.GradientBoostingRegressor()
# regressor = svm.SVR()
regressor = sken.RandomForestRegressor()
regressor.n_estimators = 16
regressor.fit(X,Y)
Ypred = regressor.predict(Xval)
print('predicted result')
np.savetxt('result_validate_quick.txt', Ypred)
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
print('score of random forest=', logscore(Ytest, Ypred))

# reg = svm.SVR()
# reg.fit(X,Y)
# regY = reg.predict(Xval)
# np.savetxt('validate_SVR.csv', regY)

scorefun = skmet.make_scorer(logscore)
# scores = skcv.cross_val_score(regressor, X, Y, scoring=scorefun, cv=5)
# print('C-V score =', np.mean(scores), '+/-', np.std(scores))

# regressor_svr = svm.SVR()
regressor_svr = sken.RandomForestRegressor()
regressor_svr.n_estimators = 32
param_grid = {'max_features': np.arange(3,32)}
neg_scorefun = skmet.make_scorer(lambda x, y: -logscore(x, y))
grid_search = skgs.GridSearchCV(regressor_svr, param_grid, scoring=neg_scorefun, cv=5)
grid_search.fit(Xtrain, Ytrain)

best = grid_search.best_estimator_
print(best)
print('best score =', -grid_search.best_score_)

#Print result to file
Ypred = best.predict(Xval)
np.savetxt('result_validate.txt', Ypred)