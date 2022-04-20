import numpy as np
import toolbox_02450 as tb
import torch
import sklearn.linear_model as lm
from data_preparation import *

# Use GPU for ANN if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

shuffled_indices = np.arange(N)
np.random.shuffle(shuffled_indices)
split = int(N / 2)

train_index = shuffled_indices[:split]
test_index = shuffled_indices[split:]

X_train = X[train_index, :]
y_train = y[train_index]

X_test = X[test_index, :]
y_test = y[test_index]

mean = np.mean(X, 0)
st_dev = np.std(X, 0)
X_train_normalized = (X_train - mean) / st_dev
X_test_normalized = (X_test - mean) / st_dev

models = ["LR", "ANN", "baseline"]
n_models = len(models)
predictions = {model: None for model in models}

predictions["baseline"] = np.repeat(np.max(y_test), np.size(y_test))

best_lambda = 0.379

logistic_model = lm.LogisticRegression(C=1 / best_lambda,
                                       penalty="l2",
                                       max_iter=250,
                                       n_jobs=8,
                                       tol=1e-3)

logistic_model.fit(X_train_normalized, y_train)
predictions["LR"] = logistic_model.predict(X_test_normalized)

best_hu = 18

ANN_model = lambda: torch.nn.Sequential(torch.nn.Linear(M, best_hu).to(device),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(best_hu, best_hu).to(device),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(best_hu, C).to(device),
                                        torch.nn.Softmax(dim=1))

net, _, _ = tb.train_neural_net(model=ANN_model,
                                loss_fn=torch.nn.CrossEntropyLoss(),
                                X=torch.tensor(X_train, dtype=torch.float).to(device),
                                y=torch.tensor(y_train, dtype=torch.long).to(device),
                                n_replicates=1,
                                max_iter=10000,
                                tolerance=1e-8)

ANN_softmax_logits_test = net(torch.tensor(X_test, dtype=torch.float).to(device))
predictions["ANN"] = (torch.max(ANN_softmax_logits_test, dim=1)[1]).data.cpu().numpy()

pairwise_thetahat = np.zeros((n_models, n_models))
pairwise_lower_CI = np.zeros((n_models, n_models))
pairwise_upper_CI = np.zeros((n_models, n_models))
pairwise_pval = np.zeros((n_models, n_models))

for i, model_1 in enumerate(models):
    for j, model_2 in enumerate(models):
        if j > i:
            print(y_test)
            print(f"{model_1=}")
            print(predictions[model_1])
            print(f"{model_2=}")
            print(predictions[model_2])
            theta_hat, CI, pval = tb.mcnemar(y_test,
                                             predictions[model_1],
                                             predictions[model_2],
                                             alpha=0.05)
            pairwise_thetahat[i, j] = theta_hat
            pairwise_lower_CI[i, j] = CI[0]
            pairwise_upper_CI[i, j] = CI[1]
            pairwise_pval[i, j] = pval

print("Pairwise theta hat:")
print(pairwise_thetahat)
print("Pairwise upper confidence intervals:")
print(pairwise_upper_CI)
print("Pairwise lower confidence intervals:")
print(pairwise_lower_CI)
print("Pairwise p values:")
print(pairwise_pval)

