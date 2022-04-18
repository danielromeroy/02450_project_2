# Linear regression, part a, section 1

# Loading libraries
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show
import sklearn.linear_model as lm
from codon_usage_load import *

## Feature transformation
# calculate the average for each column/codon
X_means = X[0,:] * 0 # array of 64 zeroes
for j in range(len(X_means)):
    X_means[j] = X[:,j].mean()
# subtract it from each value in the appropriate column
for i in range(len(X)):
    for j in range(len(X_means)):
        X[i,j] = X[i,j] - X_means[j]

# calculate the standard deviation for each column
X_stddev = X[0,:] * 0
for j in range(len(X_stddev)):
    X_stddev[j] = X[:,j].std()
# divide by the standard deviation to get std.dev = 1
for i in range(len(X)):
    for j in range(len(X_means)):
        X[i,j] = X[i,j] / X_stddev[j]

# Split dataset into features and target vector
target = "AUG"
codon_idx = attributeNames.tolist().index(target)
y = X[:,codon_idx]

X_cols = list(range(0,codon_idx)) + list(range(codon_idx+1,len(attributeNames)))
X = X[:,X_cols]

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict codon frequency
y_est = model.predict(X)
residual = y_est-y

# Display scatter plot
figure()
subplot(2,1,1)
plot(y, y_est, '.')
xlabel('{0} codon frequency (true)'.format(target)); ylabel('{0} codon frequency (estimated)'.format(target));
subplot(2,1,2)
hist(residual,40)

show()