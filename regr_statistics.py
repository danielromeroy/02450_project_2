import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate, train_neural_net
from scipy.stats import mode
import scipy.stats as st
import torch
from data_preparation import *

target = "AUG"
attributeNames = attributeNames.tolist()
codon_idx = attributeNames.index(target)
y = X[:, codon_idx]
X_cols = list(range(0, codon_idx)) + list(range(codon_idx + 1, len(attributeNames)))
X = X[:, X_cols]
M = M - 1

X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = [u'Offset'] + attributeNames
M = M + 1

# Initialize variables
# T = len(lambdas)
K = 1
lambdas = np.array([np.power(10., -4)])
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

k = 0
# Partition data into training and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

# Compute optimal lambda
internal_cross_validation = 10
opt_val_err, opt_lambdas[k], mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(
    X_train, y_train, lambdas, internal_cross_validation)

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

best_hu = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# Define the  best model
best_model = lambda: torch.nn.Sequential(
    torch.nn.Linear(M, best_hu).to(device),  # M features to n_hidden_units
    torch.nn.Tanh(),  # 1st transfer function,
    torch.nn.Linear(best_hu, 1).to(device),  # n_hidden_units to 1 output neuron
    # no final tranfer function, i.e. "linear output"
)
loss_fn = torch.nn.MSELoss()  # mean-squared-error loss

# Extract training and test set for current CV fold, convert to tensors
X_train_ANN = torch.Tensor(X_train).to(device)
y_train_ANN = torch.Tensor(y_train).to(device)
X_test_ANN = torch.Tensor(X_test).to(device)
y_test_ANN = torch.Tensor(y_test).to(device)

# Train the best model
net, final_loss, learning_curve = train_neural_net(best_model,
                                                   loss_fn,
                                                   X=X_train_ANN,
                                                   y=y_train_ANN,
                                                   n_replicates=1,
                                                   max_iter=15000)

# Calculate the new errors for the best model

# Determine estimated value
y_test_est_ANN = net(X_test_ANN).cpu().detach().numpy().T.squeeze()

# Compute z for the models
z_nofeatures = np.abs(y_test - y_test.mean()) ** 2
z_noreg = np.abs(y_test - X_test @ w_noreg[:, k]) ** 2
z_rlr = np.abs(y_test - X_test @ w_rlr[:, k]) ** 2
z_ann = np.abs(y_test - y_test_est_ANN) ** 2



# Compute the pairwise differences in z for the models
z_ann_rlr = z_ann - z_rlr
z_ann_nor = z_ann - z_noreg
z_ann_nof = z_ann - z_nofeatures
z_rlr_nor = z_rlr - z_noreg
z_rlr_nof = z_rlr - z_nofeatures
z_nor_nof = z_noreg - z_nofeatures
# save the z = zA-zB values for each outer fold

# Calculate confidence intervals and p values
alpha = 0.95
# ANN vs. regularized linear regression
p_ann_rlr = 2 * st.t.cdf(-np.abs(np.mean(z_ann_rlr)) / st.sem(z_ann_rlr), df=len(z_ann_rlr) - 1)  # p-value
CI_ann_rlr = st.t.interval(1 - alpha, len(z_ann_rlr) - 1, loc=np.mean(z_ann_rlr),
                           scale=st.sem(z_ann_rlr))  # Confidence interval
# ANN vs. non-regularized linear regression
p_ann_nor = 2 * st.t.cdf(-np.abs(np.mean(z_ann_nor)) / st.sem(z_ann_nor), df=len(z_ann_nor) - 1)  # p-value
CI_ann_nor = st.t.interval(1 - alpha, len(z_ann_nor) - 1, loc=np.mean(z_ann_nor),
                           scale=st.sem(z_ann_nor))  # Confidence interval
# ANN vs. baseline
p_ann_nof = 2 * st.t.cdf(-np.abs(np.mean(z_ann_nof)) / st.sem(z_ann_nof), df=len(z_ann_nof) - 1)  # p-value
CI_ann_nof = st.t.interval(1 - alpha, len(z_ann_nof) - 1, loc=np.mean(z_ann_nof),
                           scale=st.sem(z_ann_nof))  # Confidence interval
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
print("Artificial neural networks vs. regularized linear regression:")
print("- p-value:\t\t\t\t{}".format(p_ann_rlr))
print("- Confidence interval:\t[{};{}]".format(CI_ann_rlr[0], CI_ann_rlr[1]))
print("Artificial neural networks vs. non-regularized linear regression:")
print("- p-value:\t\t\t\t{}".format(p_ann_nor))
print("- Confidence interval:\t[{};{}]".format(CI_ann_nor[0], CI_ann_nor[1]))
print("Artificial neural networks vs. baseline:")
print("- p-value:\t\t\t\t{}".format(p_ann_nof))
print("- Confidence interval:\t[{};{}]".format(CI_ann_nof[0], CI_ann_nof[1]))
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
