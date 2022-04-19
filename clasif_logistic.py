from data_preparation import *
import numpy as np
import sklearn.linear_model as lm
from sklearn import model_selection as ms
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime as dt
from data_preparation import *

K_FOLDS = 10

lambda_interval = np.logspace(-8, 3, 20)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))

cross_validation = ms.KFold(K_FOLDS, shuffle=True)

for (k, (train_index, test_index)) in enumerate(cross_validation.split(X, y)):
    print(f"Starting fold {k + 1} of {K_FOLDS}. {round(k/K_FOLDS * 100, 1)}% done.")

    X_train = X[train_index, :]
    X_test = X[test_index, :]

    y_train = y[train_index]
    y_test = y[test_index]

    mean = np.mean(X_train, 0)
    st_dev = np.std(X_train, 0)

    X_train = (X_train - mean) / st_dev
    X_test = (X_test - mean) / st_dev

    print(f"{np.mean(X_train)=}")
    print(f"{np.std(X_train)=}")

    for i, reg_lambda in enumerate(lambda_interval):
        logistic_model = lm.LogisticRegression(C=1/reg_lambda,
                                               penalty="l2",
                                               max_iter=200)  # Increase max_iter, as it otherwise wouldn't converge

        logistic_model.fit(X_train, y_train)

        y_train_est = logistic_model.predict(X_train)  # Prediction
        y_test_est = logistic_model.predict(X_test)  # Prediction

        train_errors = np.sum(y_train_est != y_train)
        test_errors = np.sum(y_test_est != y_test)

        train_error_rate = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate = np.sum(y_test_est != y_test) / len(y_test)

        print(f"Model: Lambda =   {reg_lambda}")
        print(f"Train error rate: {(train_error_rate * 100).__round__(3)}%")
        print(f"Test error rate:  {(test_error_rate * 100).__round__(3)}%")
        print(f"Train accuracy:   {((1 - train_error_rate) * 100).__round__(3)}%")
        print(f"Test accuracy:    {((1 - test_error_rate) * 100).__round__(3)}%")

        # Train ANN with i + 1 hidden units in all

    frequency_of_most_common = np.max(np.bincount(y_test))
    baseline_accuracy = frequency_of_most_common / len(y_test)
    print(f"Accuracy of baseline: {(baseline_accuracy * 100).__round__(5)}%")



#
# # Wrangle the data bit over here
#
# #####
#
# # TODO: Add regularization param
#
# logistic_model = lm.LogisticRegression(max_iter=200)  # Increase max_iter, as it wouldn't converge otherwise
# logistic_model.fit(X, y)  # Multiclass, not binary
#
# y_est = logistic_model.predict(X)  # Prediction
#
# correct_predictions = np.sum(y == y_est)
#
# frequency_of_most_common = np.max(np.bincount(y))
#
# baseline_accuracy = frequency_of_most_common / N
# print(f"Accuracy of baseline: {(baseline_accuracy * 100).__round__(2)}%")
#
# logistic_acuracy = correct_predictions / N
# print(f"Accuracy of model:    {(logistic_acuracy * 100).__round__(2)}%")
