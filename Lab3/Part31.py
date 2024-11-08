# Import all the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
# Import the Boston dataset

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X = data
Y= target
cv = 10 # 10-fold cross-validation

# Reshuffle the data
'''
np.random.seed(0)
indices = np.random.permutation(len(X))
X = X[indices]
Y = Y[indices]
'''


print("\nLinear Regression")
lin = LinearRegression()
score = cross_val_score(lin, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(lin, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nRidge Regression")
ridge = Ridge(alpha=1.0)
score = cross_val_score(ridge, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(ridge, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nLasso Regression")
lasso = Lasso(alpha=0.1)
score = cross_val_score(lasso, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(lasso, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nDecision Tree regression")
tree = DecisionTreeRegressor(random_state=0)
score = cross_val_score(tree, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(tree, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nRandom Forest regression")
forest = RandomForestRegressor(n_estimators=50, max_depth=None, 
                               min_samples_split=2, random_state=0)
score = cross_val_score(forest, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(forest, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nLinear Support Vector Machine regression")
svm_lin = svm.SVR(epsilon=0.2, kernel='linear', C=1)
score = cross_val_score(svm_lin, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(svm_lin, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nSupport Vector Machine regression rbf")
clf = svm.SVR(epsilon=0.2, kernel='rbf', C=1)
score = cross_val_score(clf, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(clf, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nK-Nearest Neighbors regression")
knn = KNeighborsRegressor()
score = cross_val_score(knn, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(knn, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

# RFE
from sklearn.feature_selection import RFE
best_features=4

print("\nLinear Regression with RFE")
rfe_lin = RFE(lin,step=best_features).fit(X, Y)
mask = np.array(rfe_lin.support_)
score = cross_val_score(lin, X[:, mask], Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(lin, X[:, mask], Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nRidge Regression with RFE")
rfe_ridge = RFE(ridge,step=best_features).fit(X, Y)
mask = np.array(rfe_ridge.support_)
score = cross_val_score(ridge, X[:, mask], Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(ridge, X[:, mask], Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nLasso Regression with RFE")
rfe_lasso = RFE(lasso,step=best_features).fit(X, Y)
mask = np.array(rfe_lasso.support_)
score = cross_val_score(lasso, X[:, mask], Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(lasso, X[:, mask], Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nDecision Tree regression with RFE")
rfe_tree = RFE(tree, step=best_features).fit(X, Y)
mask = np.array(rfe_tree.support_)
score = cross_val_score(tree, X[:, mask], Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(tree, X[:, mask], Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nRandom Forest regression with RFE")
rfe_forest = RFE(forest, step=best_features).fit(X, Y)  
mask = np.array(rfe_forest.support_)
score = cross_val_score(forest, X[:, mask], Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(forest, X[:, mask], Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))

print("\nLinear Support Vector Machine regression with RFE")
rfe_svm_lin = RFE(svm_lin, step=best_features).fit(X, Y)
mask = np.array(rfe_svm_lin.support_)
score = cross_val_score(svm_lin, X[:, mask], Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
predicted = cross_val_predict(svm_lin, X[:, mask], Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))