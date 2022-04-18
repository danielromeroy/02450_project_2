from data_preparation import *
import numpy as np
from sklearn import model_selection as ms
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary
import torch
import matplotlib.pyplot as plt
from scipy import stats

TEST_SUBSET = 1000

N = TEST_SUBSET if TEST_SUBSET is not None else N

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

indices = np.arange(N)
np.random.shuffle(indices)

cut = int(N * 0.25)

i_test = indices[:cut]
i_train = indices[cut:]

X_test = X[i_test,:]
X_train = X[i_train,:]

y_test = y[i_test]
y_train = y[i_train]

# y = y.reshape(-1,1)
# print(y)



n_hidden_units = 11 # number of hidden units in the signle hidden layer

model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, n_hidden_units).to(device), #M features to H hiden units
                            torch.nn.ReLU(), # transfer function
                            torch.nn.Linear(n_hidden_units, n_hidden_units).to(device), #M features to H hiden units
                            torch.nn.ReLU(), # transfer function
                            torch.nn.Linear(n_hidden_units, C).to(device), # C logits
                            torch.nn.Softmax(dim=1) # final tranfer function
                            )

loss_fn = torch.nn.CrossEntropyLoss()

net, _, _ = train_neural_net(model, loss_fn,
                             X=torch.tensor(X_train, dtype=torch.float).to(device),
                             y=torch.tensor(y_train, dtype=torch.long).to(device),
                             n_replicates=3,
                             max_iter=10000)

softmax_logits = net(torch.tensor(X_test, dtype=torch.float).to(device))
y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.cpu().numpy()
e = (y_test_est != y_test)
print(f'Number of miss-classifications for ANN:\n\t {sum(e)} out of {len(e)}.\n\tError rate: {sum(e)/len(e)}'
      f'\n\tAccuracy: {((1 - sum(e)/len(e)) * 100).__round__(2)}%')
