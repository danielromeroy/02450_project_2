# # Linear regression, part a, section 2
# L2 normalisation / ridge regression

# Loading libraries
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from codon_usage_load import *
import sklearn.linear_model as lm

# I don't think it makes sense to add an offset
# The benefits of offsetting is that it allows features with small values
# to still provide a good fit for the data.
# Since all of our features are of the same type (codon frequencies),
# this should not be necessary.
# More info: https://web.mit.edu/zoya/www/linearRegression.pdf
'''
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1
'''


# Values of lambda
lambdas = np.power(10., range(0,11))

# Initialise variables
K = 1 # just train one model
k = 0
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M))
sigma = np.empty((K, M))
w_noreg = np.empty((M,K))

# Partition data into training and test sets
t = 0.1 # Ratio of test set size to full data set size
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = t, random_state = 42)
internal_cross_validation = 10

# Compute optimal lambda
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

## Feature transformation
# Standardize data based on training set, and save the mean and standard
# deviations since they're part of the model
# Calculate the mean and standard deviation for each column/codon
# Subtract the means from each value in the appropriate column
# and divide by the standard deviation to get std.dev = 1 and mean = 0
mu[k, :] = np.mean(X_train, 0)
sigma[k, :] = np.std(X_train, 0)
X_train = (X_train - mu[k, :] ) / sigma[k, :] 
X_test = (X_test - mu[k, :] ) / sigma[k, :] 


### The rest of the document is copy-pasted from ex8_1_1.py
# will look into it later
# don't know why we're transposing here... Will find out later.
Xty = X_train.T @ y_train
XtX = X_train.T @ X_train

# Compute mean squared error without using the input data at all
Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

# Estimate weights for the optimal value of lambda, on entire training set
lambdaI = opt_lambda * np.eye(M)
lambdaI[0,0] = 0 # Do no regularize the bias term
w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()

# Compute mean squared error with regularization with optimal lambda
Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

# Estimate weights for unregularized linear regression, on entire training set
w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
# Compute mean squared error without regularization
Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
# OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
#m = lm.LinearRegression().fit(X_train, y_train)
#Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
#Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

# Display the results
figure(k, figsize=(12,8))
subplot(1,2,1)
semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
# You can choose to display the legend, but it's omitted for a cleaner 
# plot, since there are many attributes
#legend(attributeNames[1:], loc='best')

subplot(1,2,2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()

# To inspect the used indices, use these print statements
#print('Cross validation fold {0}/{1}:'.format(k+1,K))
#print('Train indices: {0}'.format(train_index))
#print('Test indices: {0}\n'.format(test_index))

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))

print('Ran Exercise 8.1.1')