# # Linear regression, part a, section 2
# L2 normalisation / ridge regression

# Loading libraries
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                              title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate, train_neural_net
from data_preparation import *
from scipy.stats import mode
import scipy.stats as st
import torch

# Split dataset into features and target vector
target = "AUG"
attributeNames = attributeNames.tolist()
codon_idx = attributeNames.index(target)
y = X[:, codon_idx]
X_cols = list(range(0, codon_idx)) + list(range(codon_idx + 1, len(attributeNames)))
X = X[:, X_cols]
M = M - 1

# I am going to rewrite the script so it will do crossvalidation of the model
# in an outer fold (K = 5 or K = 10) while still finding the optimal lambda
# for each iteration and estimating the test/generalization error
# The end result will be an average of all the validated models.
# ^ Successfully did it. Now, I just need to add functionality:
# Choose the average/median opt lambda (atm the script just chooses the last)
# Which isn't realy a problem, as the lambdas are all 10^2.
# The plot title looks silly, tho.


# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = [u'Offset'] + attributeNames
M = M + 1
# I originally didnt think it made sense to add an offset
# The benefits of offsetting is that it allows features with small values
# to still provide a good fit for the data.
# Since all of our features are of the same type (codon frequencies),
# this should not be necessary.
# More info: https://web.mit.edu/zoya/www/linearRegression.pdf
# But, adding an offset changed the test error from ~36 to ~ 5.5
# And the R^2 for both train and test to go from -1.9 to 0.56-0.60.


## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True, random_state=42)
# CV = model_selection.KFold(K, shuffle=False)
# Values of lambda
lambdas = np.power(10., range(-4, 9))
# lambdas = np.power(10., (-1000,-100,-10,-5,-2))
# lambdas = np.power(10., (-100,-10,-5,-2,-1,1,2,5,10,100))


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
opt_lambdas = np.empty((K))
train_err_vs_lambda = np.empty((len(lambdas), K))
test_err_vs_lambda = np.empty((len(lambdas), K))

k = 0
# Partition data into training and test sets
for train_index, test_index in CV.split(X, y):
    # extract training and test set for current CV fold
    if len(test_index) == 1303:
        # move one observation to the train_index
        # it ensures that the split is the same for each fold
        # (10% of 13,026 is an uneven number, and thus the split varies between folds)
        train_index = np.append(train_index, [test_index[-1]], 0)
        test_index = np.delete(test_index, -1, 0)
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10
    # Compute optimal lambda
    opt_val_err, opt_lambdas[k], mean_w_vs_lambda, train_err_vs_lambda[:, k], test_err_vs_lambda[:, k] = rlr_validate(
        X_train, y_train, lambdas, internal_cross_validation)
    if k == 0:
        means_w_vs_lambda = mean_w_vs_lambda
    elif k == 1:
        means_w_vs_lambda = np.concatenate(([means_w_vs_lambda], [mean_w_vs_lambda]))
    else:
        means_w_vs_lambda = np.concatenate((means_w_vs_lambda, [mean_w_vs_lambda]))
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
    lambdaI = opt_lambdas[k] * np.eye(M)
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

    # Compute z for the models
    z_nofeatures = np.abs(y_test - y_test.mean()) ** 2
    z_noreg = np.abs(y_test - X_test @ w_noreg[:, k]) ** 2
    z_rlr = np.abs(y_test - X_test @ w_rlr[:, k]) ** 2
    #    z_ann =

    # Compute the pairwise differences in z for the models 
    z_rlr_nor = z_rlr - z_noreg
    z_rlr_nof = z_rlr - z_nofeatures
    z_nor_nof = z_noreg - z_nofeatures
    # save the z = zA-zB values for each outer fold
    if k == 1:
        z_rlr_nor_multi = np.concatenate(([z_rlr_nor], [z_rlr_nor]))
        z_rlr_nof_multi = np.concatenate(([z_rlr_nof], [z_rlr_nof]))
        z_nor_nof_multi = np.concatenate(([z_nor_nof], [z_nor_nof]))
    elif k > 1:
        z_rlr_nor_multi = np.concatenate((z_rlr_nor_multi, [z_rlr_nor]))
        z_rlr_nof_multi = np.concatenate((z_rlr_nof_multi, [z_rlr_nof]))
        z_nor_nof_multi = np.concatenate((z_nor_nof_multi, [z_nor_nof]))

        # do code here
    k += 1

# Averaging across the simulations/crossvalidations
z_rlr_nor = z_rlr_nor_multi.mean(0)
z_rlr_nof = z_rlr_nof_multi.mean(0)
z_nor_nof = z_nor_nof_multi.mean(0)

# Calculate confidence intervals and p values
alpha = 0.95
# Regularized linear regression vs. non-regularized linear regression
p_rlr_nor = 2 * st.t.cdf(-np.abs(np.mean(z_rlr_nor)) / st.sem(z_rlr_nor), df=len(z_rlr_nor) - 1)  # p-value
CI_rlr_nor = st.t.interval(1 - alpha, len(z_rlr_nor) - 1, loc=np.mean(z_rlr_nor),
                           scale=st.sem(z_rlr_nor))  # Confidence interval
# Regularized linear regression vs. baseline
p_rlr_nof = 2 * st.t.cdf(-np.abs(np.mean(z_rlr_nof)) / st.sem(z_rlr_nof), df=len(z_rlr_nof) - 1)  # p-value
CI_rlr_nof = st.t.interval(1 - alpha, len(z_rlr_nof) - 1, loc=np.mean(z_rlr_nof),
                           scale=st.sem(z_rlr_nof))  # Confidence interval
# Non-regularized linear regression vs. baseline
p_nor_nof = 2 * st.t.cdf(-np.abs(np.mean(z_nor_nof)) / st.sem(z_nor_nof), df=len(z_nor_nof) - 1)  # p-value
CI_nor_nof = st.t.interval(1 - alpha, len(z_nor_nof) - 1, loc=np.mean(z_nor_nof),
                           scale=st.sem(z_nor_nof))  # Confidence interval

# Print the results
print("Regularized linear regression vs. non-regularized linear regression:")
print("- p-value:\t\t\t\t{}".format(p_rlr_nor))
print("- Confidence interval:\t[{};{}]".format(CI_rlr_nor[0], CI_rlr_nor[1]))
print("Regularized linear regression vs. baseline:")
print("- p-value:\t\t\t\t{}".format(p_rlr_nof))
print("- Confidence interval:\t[{};{}]".format(CI_rlr_nof[0], CI_rlr_nof[1]))
print("Non-regularized linear regression vs. baseline:")
print("- p-value:\t\t\t\t{}".format(p_nor_nof))
print("- Confidence interval:\t[{};{}]".format(CI_nor_nof[0], CI_nor_nof[1]))

# Averaging across the simulations/crossvalidations
mean_w_vs_lambda = means_w_vs_lambda.mean(0)
test_err_vs_lambda = test_err_vs_lambda.mean(1)
train_err_vs_lambda = train_err_vs_lambda.mean(1)
opt_lambda = mode(opt_lambdas)[0][0]
print("\n")
# Display results
print("Optimal lambda: 1e{}".format(np.log10(opt_lambda)))
print("Linear regression without feature selection:")
print('- Training error: {0}'.format(Error_train_nofeatures.mean()))
print('- Test error:     {0}'.format(Error_test_nofeatures.mean()))
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
print(
    '- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum() - Error_test_rlr.sum()) / Error_test_nofeatures.sum()))
