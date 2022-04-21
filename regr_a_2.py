# # Linear regression, part a, section 2
# L2 normalisation / ridge regression

# Loading libraries
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                              title, subplot, show, grid, suptitle, savefig)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from data_preparation import *
from scipy.stats import mode

# Split dataset into features and target vector
target = "AUG"
attributeNames = attributeNames.tolist()
codon_idx = attributeNames.index(target)
y = X[:, codon_idx]
X_cols = list(range(0, codon_idx)) + list(range(codon_idx + 1, len(attributeNames)))
X = X[:, X_cols]
M = M - 1

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = [u'Offset'] + attributeNames
M = M + 1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 1  # just make one model. No double cross-validation in this script.
# CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambdas = np.power(10., range(-4, 9))
# lambdas = np.power(10., range(6,18))

# Initialize variables
# T = len(lambdas)
Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_rlr = np.empty((K, 1))
Error_test_rlr = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))
w_rlr = np.empty((M, K))
mu = np.empty((K, M - 1))
sigma = np.empty((K, M - 1))
w_noreg = np.empty((M, K))

k = 0
# Partition data into training and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

# Compute optimal lambda   
internal_cross_validation = 10
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train,
                                                                                                  lambdas,
                                                                                                  internal_cross_validation)

## Feature transformation
# Standardize data based on training set, and save the mean and standard
# deviations since they're part of the model
# Calculate the mean and standard deviation for each column/codon
# Subtract the means from each value in the appropriate column
# and divide by the standard deviation to get std.dev = 1 and mean = 0
mu[k, :] = np.mean(X_train[:, 1:], 0)
sigma[k, :] = np.std(X_train[:, 1:], 0)
X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

# Transpose X_train and (matrix) multiply with y_train or X_train
Xty = X_train.T @ y_train
XtX = X_train.T @ X_train

# Compute mean squared error without using the input data at all
Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
Error_test_nofeatures[k] = np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]

# Estimate weights for the optimal value of lambda, on entire training set
lambdaI = opt_lambda * np.eye(M)
lambdaI[0, 0] = 0  # Do no regularize the bias term
w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
# Compute mean squared error with regularization with optimal lambda
Error_train_rlr[k] = np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
Error_test_rlr[k] = np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]

# Estimate weights for unregularized linear regression, on entire training set
w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
# Compute mean squared error without regularization
Error_train[k] = np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]
Error_test[k] = np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]

# Plot the results (averaged)
figure(k, figsize=(12, 8))
subplot(1, 2, 1)
semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], '.-')  # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
# You can choose to display the legend, but it's omitted for a cleaner 
# plot, since there are many attributes
# legend(attributeNames[1:], loc='best')

subplot(1, 2, 2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas, train_err_vs_lambda.T, 'b.-', lambdas, test_err_vs_lambda.T, 'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Training error', 'Generalization error'])
grid()
suptitle("Linear regression with regularization",
         fontsize=22,
         fontweight="bold")
savefig("../results/reg_2.png")

# Display results
print('Linear regression without regularization:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum() - Error_train.sum()) / Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format(
    (Error_train_nofeatures.sum() - Error_train_rlr.sum()) / Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum() - Error_test_rlr.sum()) / Error_test_nofeatures.sum()))

# print('Weights in last fold:')
# for m in range(M):
#    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))
