'''
Created on Mar 7, 2015

@author: jonathan
'''

if __name__ == '__main__':
    pass

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

def get_features(h):
    return [h, np.exp(h)]


def read_data(inpath):
    X = []
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            X.append(get_features(t.hour))
    return np.atleast_2d(X)


X = read_data('train.csv')
Y = np.genfromtxt('train_y.csv', delimiter=',')
print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)

Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.75)
print('Shape of Xtrain:', Xtrain.shape)
print('Shape of Ytrain:', Ytrain.shape)
print('Shape of Xtest:', Xtest.shape)
print('Shape of Ytest:', Ytest.shape)

regressor = sklin.LinearRegression()
regressor.fit(Xtrain, Ytrain)
print('coefficients =', regressor.coef_)
print('intercept =', regressor.intercept_)

# plt.plot(Xtrain[:, 0], Ytrain, 'bo')
# plt.xlim([-0.5, 23.5])
# plt.ylim([0, 1000])
# plt.show()

Hplot = range(25)
Xplot = np.atleast_2d([get_features(x) for x in Hplot])
Yplot = regressor.predict(Xplot)
plt.plot(Xtrain[:, 0], Ytrain, 'bo')
plt.plot(Hplot, Yplot, 'r', linewidth=3)
plt.xlim([-0.5, 23.5])
plt.ylim([0, 1000])
plt.show()

def logscore(gtruth, pred):
    pred = np.clip(pred, 0, np.inf)
    logdif = np.log(1 + gtruth) - np.log(1 + pred)
    return np.sqrt(np.mean(np.square(logdif)))


Ypred = regressor.predict(Xtest)
print('score =', logscore(Ytest, Ypred))

scorefun = skmet.make_scorer(logscore)
scores = skcv.cross_val_score(regressor, X, Y, scoring=scorefun, cv=5)
print('C-V score =', np.mean(scores), '+/-', np.std(scores))

regressor_ridge = sklin.Ridge()
param_grid = {'alpha': np.linspace(0, 100, 10)}
neg_scorefun = skmet.make_scorer(lambda x, y: -logscore(x, y))
grid_search = skgs.GridSearchCV(regressor_ridge, param_grid, scoring=neg_scorefun, cv=5)
grid_search.fit(Xtrain, Ytrain)

best = grid_search.best_estimator_
print(best)
print('best score =', -grid_search.best_score_)

#Print result to file
Xval = read_data('validate.csv')
Ypred = best.predict(Xval)
np.savetxt('result_validate.txt', Ypred)