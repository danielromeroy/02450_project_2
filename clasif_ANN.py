import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary
from scipy import stats
from datetime import datetime as dt
from data_preparation import *

TEST_SUBSET = None
K_FOLDS = 10

if TEST_SUBSET is not None:
    X = X[:TEST_SUBSET,]
    y = y[:TEST_SUBSET]
    N = TEST_SUBSET

# Use GPU if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

indices = np.arange(N)
np.random.shuffle(indices)

cross_validation = ms.KFold(K_FOLDS, shuffle=True)

n_hidden_units = range(1, 16) # Number of hidden units to test
results = []
for (k, (train_index, test_index)) in enumerate(cross_validation.split(X, y)):
    print(f"Starting fold {k + 1} of {K_FOLDS}. {round(k/K_FOLDS, 1)}% done.")
    for hu in n_hidden_units:
        print(f"Training ANN with {hu} hiden units in fold {k + 1}.")
        X_train = X[train_index,:]
        X_test = X[test_index,:]

        y_train = y[train_index]
        y_test = y[test_index]

        model = lambda: torch.nn.Sequential(torch.nn.Linear(M, hu).to(device), #M features to H hiden units
                                            torch.nn.ReLU(), # transfer function
                                            torch.nn.Linear(hu, hu).to(device), #M features to H hiden units
                                            torch.nn.ReLU(), # transfer function
                                            torch.nn.Linear(hu, C).to(device), # C logits
                                            torch.nn.Softmax(dim=1)) # final tranfer function

        loss_fn = torch.nn.CrossEntropyLoss()

        net, _, _ = train_neural_net(model,
                                     loss_fn,
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
              f'\tError rate: {sum(e)/len(e)}\n'
              f'\tAccuracy: {((1 - sum(e)/len(e)) * 100).__round__(2)}%')
        # Array keys: fold | hidden units | errors | test set size | error rate | accuracy
        result_array = np.array([k + 1, hu, sum(e), len(e), sum(e)/len(e), 1 - sum(e)/len(e)])
        results.append(result_array)

np.set_printoptions(suppress=True, linewidth=np.inf)
results.insert(0, np.array(["k-fold", "hidden units", "n errors", "test set size", "error rate", "accuracy"]))
results_array = np.vstack(tuple(results))
print(results_array)

out_file = f"ANN_performance_{dt.now().strftime('%d_%m_%Y_%H_%M')}.csv"
np.savetxt(f"./results/{out_file}", results_array, delimiter=",")
