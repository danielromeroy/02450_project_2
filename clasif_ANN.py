from data_preparation import *
import numpy as np
from sklearn import model_selection as ms
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary
import torch
import matplotlib.pyplot as plt
from scipy import stats



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X = X[:4000]
y = y[:4000]

K = 2
cross_validation = ms.KFold(K, shuffle=True)

n_hidden_units = 2

model = lambda: torch.nn.Sequential(torch.nn.Linear(M, n_hidden_units).to(device),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(n_hidden_units, 1).to(device),
                                    torch.nn.Sigmoid())

max_iter = 10000

##### STUFF FROM THE SCRIPT ####

# Setup figure for display of the decision boundary for the several crossvalidation folds.
decision_boundaries = plt.figure(1, figsize=(10, 10))
# Determine a size of a plot grid that fits visualizations for the chosen number
# of cross-validation splits, if K=4, this is simply a 2-by-2 grid.
subplot_size_1 = int(np.floor(np.sqrt(K)))
subplot_size_2 = int(np.ceil(K / subplot_size_1))
# Set overall title for all of the subplots
plt.suptitle('Data and model decision boundaries', fontsize=20)
# Change spacing of subplots
plt.subplots_adjust(left=0, bottom=0, right=1, top=.9, wspace=.5, hspace=0.25)

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1, 2, figsize=(10, 5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

##############################<3


print(f'Model to train:\n{str(model())}')
errors = []
for k, (train_index, test_index) in enumerate(cross_validation.split(X, y)):
    print(f'\nCrossvalidation fold {k + 1} of {K}...')

    # Create torch tensors
    X_train = torch.Tensor(X[train_index, :]).to(device)
    y_train = torch.Tensor(y[train_index]).to(device)
    X_test = torch.Tensor(X[test_index, :]).to(device)
    y_test = torch.Tensor(y[test_index]).to(device)

    print(X_train.size())
    print(y_train.size())
    print(X_test.size())
    print(y_test.size())

    print(X_train.is_cuda)
    print(y_train.is_cuda)
    print(X_test.is_cuda)
    print(y_test.is_cuda)

    # Use toolbox function to train the ANN
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn=torch.nn.L1Loss(),
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=3,
                                                       max_iter=max_iter)

    print('\n\tBest loss: {}\n'.format(final_loss))

    y_sigmoid = net(X_test)  # activation of final note, i.e. prediction of network
    y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8)  # threshold output of sigmoidal function
    y_test = y_test.type(dtype=torch.uint8)
    # Determine errors and error rate
    e = (y_test_est != y_test)
    error_rate = (sum(e).type(torch.float) / len(y_test)).data.cpu().numpy()
    errors.append(error_rate)  # store error rate for current CV fold

    # Make a subplot for current cross validation fold that displays the
    # decision boundary over the original data, "background color" corresponds
    # to the output of the sigmoidal transfer function (i.e. before threshold),
    # white areas are areas of uncertainty, and a deaper red/blue means
    # that the network "is more sure" of a given class.
    plt.figure(decision_boundaries.number)
    plt.subplot(subplot_size_1, subplot_size_2, k + 1)
    plt.title('CV fold {0}'.format(k + 1), color=color_list[k])
    predict = lambda x: net(torch.tensor(x, dtype=torch.float)).data.numpy()
    # visualize_decision_boundary(predict, X, y,  # provide data, along with function for prediction
    #                             attributeNames, classNames,  # provide information on attribute and class names
    #                             train=train_index, test=test_index,  # provide information on partioning
    #                             show_legend=k == (K - 1))  # only display legend for last plot

    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k + 1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')

# Display the error rate across folds
# summaries_axes[1].bar(np.arange(1, K + 1), np.squeeze(np.asarray(errors)), color=color_list)
summaries_axes[1].set_xlabel('Fold');
summaries_axes[1].set_xticks(np.arange(1, K + 1))
summaries_axes[1].set_ylabel('Error rate');
summaries_axes[1].set_title('Test misclassification rates')

# Show the plots
plt.show(decision_boundaries.number) # try these lines if the following code fails (depends on package versions)
plt.show(summaries.number)
# plt.show()

# Display a diagram of the best network in last fold
print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0, 2]]
biases = [net[i].bias.data.numpy() for i in [0, 2]]
tf = [str(net[i]) for i in [1, 3]]
draw_neural_net(weights, biases, tf)

# Print the average classification error rate
print('\nGeneralization error/average error rate: {0}%'.format(round(100 * np.mean(errors), 4)))



