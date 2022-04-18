import data_preparation
import numpy as np
import sklearn.linear_model as lm

# Wrangle the data bit over here

#####

logistic_model = lm.LogisticRegression(max_iter=200)  # Increase max_iter, as it wouldn't converge otherwise
logistic_model.fit(X, y)  # Multiclass, not binary

y_est = logistic_model.predict(X)  # Prediction

correct_predictions = np.sum(y == y_est)

frequency_of_most_common = np.max(np.bincount(y))

baseline_accuracy = frequency_of_most_common / N
print(f"Accuracy of baseline: {(baseline_accuracy * 100).__round__(2)}%")

logistic_acuracy = correct_predictions / N
print(f"Accuracy of model:    {(logistic_acuracy * 100).__round__(2)}%")
