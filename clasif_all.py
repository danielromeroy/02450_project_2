import sklearn.linear_model as lm
from sklearn import model_selection as ms
import torch
from toolbox_02450 import train_neural_net
from datetime import datetime as dt
import numpy as np
from data_preparation import *

TEST = False
TEST_SUBSET = None if not TEST else 200
# K_FOLDS = 10 if not TEST else 2
K_FOLDS = 3 if not TEST else 2

if TEST_SUBSET is not None:
    shuffled_indices = np.arange(N)
    np.random.shuffle(shuffled_indices)
    test_indices = shuffled_indices[:TEST_SUBSET]
    X = X[test_indices, ]
    y = y[test_indices]
    N = TEST_SUBSET

# Use GPU for ANN if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")


def select_best_model(results: dict):
    gen_errors = {model: 0 for model in results.keys()}
    for model, results_list in results.items():
        results_array = np.vstack(tuple(results_list))
        val_set_size = np.sum(results_array[:, 1])
        gen_error = np.sum(results_array[:, 1] / val_set_size * results_array[:, 0])
        gen_errors[model] = gen_error
    gen_errors_array = np.array(list(gen_errors.values()))
    best_model_index = np.argmin(gen_errors_array)
    best_model = list(gen_errors.keys())[best_model_index]
    return best_model


def write_outfile(array, name: str, header: str):
    out_file = f"{name}_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv"
    np.savetxt(f"./results/{out_file}", array,
               delimiter=",",
               header=header,
               fmt="%.9f")


lambda_interval = np.logspace(-6, 3, 20)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))

# hidden_units_interval = np.hstack((np.arange(1, 11, 3), np.arange(12, 15, 2), np.arange(15, 21)))
hidden_units_interval = np.arange(22, 31, 2)

cross_validation = ms.KFold(K_FOLDS, shuffle=True)
outer_fold = cross_validation.split(X, y)

CV_table_results = {
    "LR": [],       # k_of, best_lambda, test_error
    "ANN": [],      # k_of, best_hu, test_error
    "baseline": []  # k_of, test_error
}

# model index: LR -> 1 | ANN -> 2 | baseline -> 0
all_data = {
    "LR": [],       # model_index, k_of, k_if, lambda, n_val_errors, Nval, val_error_rate, n_train_errors, Ntrain, train_error_rate
    "ANN": [],      # model_index, k_of, k_if, hu,     n_val_errors, Nval, val_error_rate, n_train_errors, Ntrain, train_error_rate
    "baseline": []  # model_index, k_of, k_if, 0,      n_val_errors, Nval, val_error_rate, n_train_errors, Ntrain, train_error_rate
}
for (k_of, (par_index, test_index)) in enumerate(outer_fold):
    print(f"Starting outer fold {k_of + 1} of {K_FOLDS}. {round(k_of / K_FOLDS * 100, 1)}% done.")

    X_par = X[par_index, :]
    X_test = X[test_index, :]

    y_par = y[par_index]
    y_test = y[test_index]

    mean = np.mean(X_par, 0)
    st_dev = np.std(X_par, 0)

    # Normalize for logistic regression
    X_par_normalized = (X_par - mean) / st_dev
    X_test_normalized = (X_test - mean) / st_dev

    # Compute baseline performance
    frequency_of_most_common = np.max(np.bincount(y_test))
    baseline_accuracy = frequency_of_most_common / len(y_test)
    print(f"Accuracy of baseline: {(baseline_accuracy * 100).__round__(5)}%")
    CV_table_results["baseline"].append(np.array([k_of, 1 - baseline_accuracy]))

    inner_fold = cross_validation.split(X_par, y_par)

    LR_results = {lam: [] for lam in lambda_interval}
    ANN_results = {n_hu: [] for n_hu in hidden_units_interval}
    for (k_if, (train_index, val_index)) in enumerate(inner_fold):
        print(f"Starting inner fold {k_if} of {K_FOLDS} in outer fold {k_of} of {K_FOLDS}.\n"
              f"{round(((k_of / K_FOLDS) + (1 / K_FOLDS)*(k_if / K_FOLDS)) * 100, 1)}% done.")

        X_train = X_par[train_index, :]
        X_val = X_par[val_index, :]

        X_train_normalized = X_par_normalized[train_index, :]
        X_val_normalized = X_par_normalized[val_index, :]

        y_train = y_par[train_index]
        y_val = y_par[val_index]

        frequency_of_most_common = np.max(np.bincount(y_val))
        baseline_accuracy = frequency_of_most_common / len(y_val)
        val_errors = len(y_val) - frequency_of_most_common
        val_error_rate = 1 - baseline_accuracy
        baseline_accuracy = frequency_of_most_common / len(y_train)
        train_errors = len(y_train) - frequency_of_most_common
        train_error_rate = 1 - baseline_accuracy
        all_result_array = np.array([0, k_of, k_if, 0,
                                     val_errors, len(y_val), val_error_rate,
                                     train_errors, len(y_train), train_error_rate])
        all_data["baseline"].append(all_result_array)

        # Logistic regression
        # for reg_lambda in lambda_interval:
        #     # Increase max_iter and lower tol, as it otherwise wouldn't converge
        #     logistic_model = lm.LogisticRegression(C=1 / reg_lambda,
        #                                            penalty="l2",
        #                                            max_iter=250 if not TEST else 10,
        #                                            n_jobs=8,
        #                                            tol=1e-3)
        #
        #     logistic_model.fit(X_train_normalized, y_train)
        #
        #     y_val_est = logistic_model.predict(X_val_normalized)
        #     val_errors = np.sum(y_val_est != y_val)
        #     val_error_rate = val_errors / len(y_val)
        #
        #     y_train_est = logistic_model.predict(X_train_normalized)
        #     train_errors = np.sum(y_train_est != y_train)
        #     train_error_rate = train_errors / len(y_train)
        #
        #     print(f"Model: Regularized logistic regression with lambda = {reg_lambda}\n"
        #           f"\tValidation error rate: {(val_error_rate * 100).__round__(3)}%\n"
        #           f"\tValidation accuracy:   {((1 - val_error_rate) * 100).__round__(3)}%")
        #
        #     result_array = np.array([val_error_rate, len(y_val)])  # error_rate, Nval
        #     LR_results[reg_lambda].append(result_array)
        #
        #     all_result_array = np.array([1, k_of, k_if, reg_lambda,
        #                                  val_errors, len(y_val), val_error_rate,
        #                                  train_errors, len(y_train), train_error_rate])
        #     all_data["LR"].append(all_result_array)

        # ANN
        for n_hidden_units in hidden_units_interval:
            print(f"Training ANN with {n_hidden_units} hiden units in outer fold {k_of + 1} and inner fold {k_if + 1}.")

            ANN_model = lambda: torch.nn.Sequential(torch.nn.Linear(M, n_hidden_units).to(device),  # M features to hu hiden units
                                                    torch.nn.ReLU(),  # transfer function
                                                    torch.nn.Linear(n_hidden_units, n_hidden_units).to(device),  # hu hiden units to hu hiden units
                                                    torch.nn.ReLU(),  # transfer function
                                                    torch.nn.Linear(n_hidden_units, C).to(device),  # hu hiden units to C logits
                                                    torch.nn.Softmax(dim=1))  # final transfer function

            net, _, _ = train_neural_net(model=ANN_model,
                                         loss_fn=torch.nn.CrossEntropyLoss(),
                                         X=torch.tensor(X_train, dtype=torch.float).to(device),
                                         y=torch.tensor(y_train, dtype=torch.long).to(device),
                                         n_replicates=1,  # All replicates were always the same
                                         max_iter=10000 if not TEST else 10,
                                         tolerance=1e-8)

            softmax_logits_val = net(torch.tensor(X_val, dtype=torch.float).to(device))

            y_val_est = (torch.max(softmax_logits_val, dim=1)[1]).data.cpu().numpy()
            val_errors = np.sum(y_val_est != y_val)
            val_error_rate = np.sum(y_val_est != y_val) / len(y_val)

            print(f"Model: ANN with {n_hidden_units} hidden units:\n"
                  f"\tValidation error rate:  {(val_error_rate * 100).__round__(3)}%\n"
                  f"\tValidation accuracy:    {((1 - val_error_rate) * 100).__round__(3)}%")

            softmax_logits_train = net(torch.tensor(X_train, dtype=torch.float).to(device))

            y_train_est = (torch.max(softmax_logits_train, dim=1)[1]).data.cpu().numpy()
            train_errors = np.sum(y_train_est != y_train)
            train_error_rate = np.sum(y_train_est != y_train) / len(y_train)

            result_array = np.array([val_error_rate, len(y_val)])  # error_rate, Nval
            ANN_results[n_hidden_units].append(result_array)

            all_result_array = np.array([2, k_of, k_if, n_hidden_units,
                                         val_errors, len(y_val), val_error_rate,
                                         train_errors, len(y_train), train_error_rate])
            all_data["ANN"].append(all_result_array)

    # Select best LR  based on inner CV
    # best_lambda = select_best_model(LR_results)
    # print(f"Best lambda in fold {k_of + 1}: {best_lambda}")
    # Select best ANN based on inner CV
    best_hu = select_best_model(ANN_results)
    print(f"Best hidden units in fold {k_of + 1}: {best_hu}")
    # Train best models on X_par
    # logistic_star = lm.LogisticRegression(C=1 / best_lambda,
    #                                       penalty="l2",
    #                                       max_iter=250 if not TEST else 10,
    #                                       n_jobs=8,
    #                                       tol=1e-3)

    print(f"Training logistic regression on Dpar (outer fold {k_of + 1} of {K_FOLDS}).")
    # logistic_star.fit(X_par_normalized, y_par)

    ANN_star = lambda: torch.nn.Sequential(torch.nn.Linear(M, best_hu).to(device),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(best_hu, best_hu).to(device),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(best_hu, C).to(device),
                                           torch.nn.Softmax(dim=1))

    print(f"Training ANN on Dpar (outer fold {k_of + 1} of {K_FOLDS}).")
    net, _, _ = train_neural_net(model=ANN_star,
                                 loss_fn=torch.nn.CrossEntropyLoss(),
                                 X=torch.tensor(X_par, dtype=torch.float).to(device),
                                 y=torch.tensor(y_par, dtype=torch.long).to(device),
                                 n_replicates=1,
                                 max_iter=10000 if not TEST else 10,
                                 tolerance=1e-8)

    # Calculate test error by testing newly trained models on X_test

    # LR_y_test_est = logistic_star.predict(X_test_normalized)
    # LR_test_errors = np.sum(LR_y_test_est != y_test)
    # LR_test_error_rate = LR_test_errors / len(y_test)

    # print(f"Logistic regression test error rate (outer fold {k_of} of {K_FOLDS}):\n"
    #       f"\t{LR_test_error_rate}")

    ANN_softmax_logits_test = net(torch.tensor(X_test, dtype=torch.float).to(device))
    ANN_y_test_est = (torch.max(ANN_softmax_logits_test, dim=1)[1]).data.cpu().numpy()
    ANN_test_errors = np.sum(ANN_y_test_est != y_test)
    ANN_test_error_rate = np.sum(ANN_y_test_est != y_test) / len(y_test)

    print(f"ANN test error rate (outer fold {k_of} of {K_FOLDS}):\n"
          f"\t{ANN_test_error_rate}")

    # Save results
    # CV_table_results["LR"].append(np.array([k_of, best_lambda, LR_test_error_rate]))
    CV_table_results["ANN"].append(np.array([k_of, best_hu, ANN_test_error_rate]))

np.set_printoptions(suppress=True, linewidth=np.inf)

CV_table_results_array = np.hstack((CV_table_results["LR"], CV_table_results["ANN"], CV_table_results["baseline"]))
# array keys: outer_fold, LR_best_lambda, LR_test_error, ANN_best_hu, ANN_test_error(hu), test_error(baseline)
CV_table_results_array = np.delete(CV_table_results_array, (3, 6), axis=1)

write_outfile(CV_table_results_array,
              name="2-level_CV_table",
              header="outer_fold, LR_best_lambda, LR_test_error, ANN_best_hu, ANN_test_error, baseline_test_error")

# all_result_array = np.vstack(tuple(all_data["LR"] + all_data["ANN"] + all_data["baseline"]))
all_result_array = np.vstack(tuple(all_data["ANN"]))

write_outfile(all_result_array,
              name="2lCV_all_data",
              header="model_index, k_of, k_if, complexity, n_val_errors, Nval, val_error_rate, n_train_errors, Ntrain, train_error_rate")
