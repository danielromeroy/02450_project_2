# Linear regression, part a, section 1

# Loading libraries
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show, title, savefig
import sklearn.linear_model as lm
from data_preparation import *

# Split dataset into features and target vector
target = "AUG"
attributeNames = attributeNames.tolist()
codon_idx = attributeNames.index(target)
y = X[:, codon_idx]
X_cols = list(range(0, codon_idx)) + list(range(codon_idx + 1, len(attributeNames)))
X = X[:, X_cols]

## Feature transformation
# calculate the average for each column/codon
X_means = X[0, :] * 0  # array of 64 zeroes
for j in range(len(X_means)):
    X_means[j] = X[:, j].mean()
# subtract it from each value in the appropriate column
for i in range(len(X)):
    for j in range(len(X_means)):
        X[i, j] = X[i, j] - X_means[j]

# calculate the standard deviation for each column
X_stddev = X[0, :] * 0
for j in range(len(X_stddev)):
    X_stddev[j] = X[:, j].std()
# divide by the standard deviation to get std.dev = 1
for i in range(len(X)):
    for j in range(len(X_means)):
        X[i, j] = X[i, j] / X_stddev[j]

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X, y)

# Predict codon frequency
y_est = model.predict(X)
residual = y_est - y

# Display scatter plot
figure(figsize=(10, 9))
subplot(2, 1, 1)
plot(y, y_est, '.')
xlabel('Actual {0} codon frequency'.format(target));
ylabel('Predicted {0} codon frequency'.format(target));
title("Linear regression model performance",
      pad=10,
      fontsize=20,
      fontweight="bold")
subplot(2, 1, 2)
hist(residual, 40)
xlabel("Residuals")
ylabel("Frequency")
savefig("../results/reg_1")
