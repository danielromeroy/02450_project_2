import numpy as np
import sklearn.linear_model as lm
from sklearn import model_selection as ms
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary
from datetime import datetime as dt
from data_preparation import *

TEST_SUBSET = None
K_FOLDS = 10

if TEST_SUBSET is not None:
    X = X[:TEST_SUBSET,]
    y = y[:TEST_SUBSET]
    N = TEST_SUBSET

# Use GPU for ANN if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

lambda_interval = np.logspace(-8, 3, 20)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))

cross_validation = ms.KFold(K_FOLDS, shuffle=True)

results = []
for (k, (train_index, test_index)) in enumerate(cross_validation.split(X, y)):
    print(f"Starting fold {k + 1} of {K_FOLDS}. {round(k/K_FOLDS * 100, 1)}% done.")

    X_train = X[train_index, :]
    X_test = X[test_index, :]

    y_train = y[train_index]
    y_test = y[test_index]

    mean = np.mean(X_train, 0)
    st_dev = np.std(X_train, 0)

    X_train_normalized = (X_train - mean) / st_dev
    X_test_normalized = (X_test - mean) / st_dev

    for reg_lambda in lambda_interval:
        logistic_model = lm.LogisticRegression(C=1/reg_lambda,
                                               penalty="l2",
                                               max_iter=200)  # Increase max_iter, as it otherwise wouldn't converge

        logistic_model.fit(X_train_normalized, y_train)

        y_train_est = logistic_model.predict(X_train_normalized)  # Prediction
        y_test_est = logistic_model.predict(X_test_normalized)  # Prediction

        train_errors = np.sum(y_train_est != y_train)
        test_errors = np.sum(y_test_est != y_test)

        train_error_rate = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate = np.sum(y_test_est != y_test) / len(y_test)

        print(f"Model:\n\tRegularized logistic regression with lambda = {reg_lambda}")
        print(f"Train error rate: {(train_error_rate * 100).__round__(3)}%")
        print(f"Test error rate:  {(test_error_rate * 100).__round__(3)}%")
        print(f"Train accuracy:   {((1 - train_error_rate) * 100).__round__(3)}%")
        print(f"Test accuracy:    {((1 - test_error_rate) * 100).__round__(3)}%")

    for i in range(len(lambda_interval)):
        hu = i + 1  # i + 1 hidden units

        print(f"Training ANN with {hu} hiden units in fold {k + 1}.")

        model = lambda: torch.nn.Sequential(torch.nn.Linear(M, hu).to(device),  # M features to hu hiden units
                                            torch.nn.ReLU(),  # transfer function
                                            torch.nn.Linear(hu, hu).to(device),  # hu hiden units to hu hiden units
                                            torch.nn.ReLU(),  # transfer function
                                            torch.nn.Linear(hu, C).to(device),  # hu hiden units to C logits
                                            torch.nn.Softmax(dim=1))  # final tranfer function

        loss_fn = torch.nn.CrossEntropyLoss()

        net, _, _ = train_neural_net(model, loss_fn,
                                     X=torch.tensor(X_train, dtype=torch.float).to(device),
                                     y=torch.tensor(y_train, dtype=torch.long).to(device),
                                     n_replicates=1,
                                     max_iter=15000,
                                     tolerance=1e-8)

        softmax_logits = net(torch.tensor(X_test, dtype=torch.float).to(device))
        y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.cpu().numpy()
        e = (y_test_est != y_test)
        print(f'ANN with {hu} hidden units:\n'
              f'\t{sum(e)} misclassifications out of {len(e)} tests.\n'
              f'\tError rate: {sum(e) / len(e)}\n'
              f'\tAccuracy: {((1 - sum(e) / len(e)) * 100).__round__(2)}%')
        # Array keys: fold | hidden units | errors | test set size | error rate | accuracy
        result_array = np.array([k + 1, hu, sum(e), len(e), sum(e) / len(e), 1 - sum(e) / len(e)])
        results.append(result_array)




    frequency_of_most_common = np.max(np.bincount(y_test))
    baseline_accuracy = frequency_of_most_common / len(y_test)
    print(f"Accuracy of baseline: {(baseline_accuracy * 100).__round__(5)}%")