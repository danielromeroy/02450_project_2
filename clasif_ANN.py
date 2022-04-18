from data_preparation import *
import numpy as np
from sklearn import model_selection as ms
from toolbox_02450 import train_neural_net
import torch

K = 4
cross_validation = ms.KFold(K, shuffle=True)

n_hidden_units = 4

model = lambda: torch.nn.Sequential(torch.nn.Linear(M, n_hidden_units),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(n_hidden_units, 1),
                                    torch.nn.Sigmoid())

loss_fn = torch.nn.BCELoss()  # We can put this inside the function later

max_iter = 10000

print(f'Model to train:\n{str(model())}')

for k, (train_index, test_index) in enumerate(CV.split(X, y)):
    print(f'\nCrossvalidation fold {k + 1} of {K}...')
    


