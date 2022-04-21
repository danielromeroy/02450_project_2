#!/usr/bin/env python3


# Load libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
from datetime import datetime as dt

# Load data
from data_preparation import *

# Define target and update dataset
target = "AUG"
attributeNames = attributeNames.tolist()
codon_idx = attributeNames.index(target)
y = X[:,[codon_idx]]
X_cols = list(range(0,codon_idx)) + list(range(codon_idx+1,len(attributeNames)))
X = X[:,X_cols]
M = M-1


# Subset and KFolds
TEST = True
TEST_SUBSET = None if not TEST else 200
K_FOLDS = 10 if not TEST else 3

if TEST_SUBSET is not None:
    N = TEST_SUBSET
    shuffled_indices = np.arange(N)
    np.random.shuffle(shuffled_indices)
    test_indices = shuffled_indices[:TEST_SUBSET]
    X = X[test_indices, ]
    y = y[test_indices]
    
    

# selecting the best model for different hu  
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


# write outfiles
def write_outfile(array, name: str, header: str):
    out_file = f"{name}_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv"
    np.savetxt(f"./results/{out_file}", array,
               delimiter=",",
               header=header,
               fmt="%.9f")


# Data structures for saving results

CV_table_results = {
    "ANN": []      # k_of, best_hu, test_error
}

# model index: LR -> 1 | ANN -> 2 | baseline -> 0
all_data = {
    "ANN": []      # model_index, k_of, k_if, hu,     n_val_errors, Nval, val_error_rate, n_train_errors, Ntrain, train_error_rate
}

# hidden units interval
hidden_units_interval = np.hstack((np.arange(1, 11, 3), np.arange(12, 15, 2), np.arange(15, 21)))

# Split data for outer fold cross validation 
cross_validation = model_selection.KFold(K_FOLDS, shuffle=True)
outer_fold = cross_validation.split(X, y)


## MAIN PART

# Outer fold
for (k_of, (par_index, test_index)) in enumerate(outer_fold):
    
    print(f"Starting outer fold {k_of + 1} of {K_FOLDS}. {round(k_of / K_FOLDS * 100, 1)}% done.")

    X_par = X[par_index, :]
    X_test = X[test_index, :]

    y_par = y[par_index]
    y_test = y[test_index]
    
    # Calculate mean and deviation for normalization
    
    mean = np.mean(X_par, 0)
    st_dev = np.std(X_par, 0)
    
    # Normalize for ANN regression 
    X_par_normalized = (X_par - mean) / st_dev
    X_test_normalized = (X_test - mean) / st_dev
    
    # Split data for inner fold cross validation
    inner_fold = cross_validation.split(X_par, y_par)
    
    # Inner fold
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
        
        # ANN

        for n_hidden_units in hidden_units_interval:
            
            print(f"Training ANN with {n_hidden_units} hiden units in outer fold {k_of + 1} and inner fold {k_if + 1}.")
            
            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, n_hidden_units),     # M features to n_hidden_units
                                torch.nn.Tanh(),                        # 1st transfer function,
                                torch.nn.Linear(n_hidden_units, 1),     # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
            
            # Extract training and test set for current CV fold, convert to tensors
            X_train = torch.Tensor(X[train_index,:])
            y_train = torch.Tensor(y[train_index])
            X_test = torch.Tensor(X[test_index,:])
            y_test = torch.Tensor(y[test_index])
            
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train,
                                                               y=y_train,
                                                               n_replicates=1,
                                                               max_iter=10000 if not TEST else 100)
            # print loss
            print('\n\tBest loss: {}\n'.format(final_loss))
            
            # Determine estimated value    
            y_test_est = net(X_test)
            
            # Determine errors and errors
            se = (y_test_est.float()-y_test.float())**2                     # squared error
            mse = (sum(se).type(torch.float)/len(y_test)).data.numpy()      # mean
            
            # Store errors and results
            result_array = np.array([mse, len(y_val)])                      # error_rate, Nval
            ANN_results[n_hidden_units].append(result_array)
            
            # Determine y_train_est value    
            y_train_est = net(X_train)
            
            # Determine train errors 
            se_train = (y_train_est.float()-y_train.float())**2                     # squared error
            mse_train = (sum(se_train).type(torch.float)/len(y_train)).data.numpy()      # mean
            
            # Store all results 
            
            all_result_array = np.array([1, k_of, k_if, n_hidden_units,
                                         mse, len(y_val), mse_train ])
            
            all_data["ANN"].append(all_result_array)
            

        # Select best model
        best_hu = select_best_model(ANN_results)
        print(f"Best hidden units in fold {k_of + 1}: {best_hu}")
        
        # Define the  best model
        best_model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, best_hu),     # M features to n_hidden_units
                            torch.nn.Tanh(),                 # 1st transfer function,
                            torch.nn.Linear(best_hu, 1),     # n_hidden_units to 1 output neuron
                                                             # no final tranfer function, i.e. "linear output"
                            )
        loss_fn = torch.nn.MSELoss()                         # mean-squared-error loss
        
        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index,:])
        y_test = torch.Tensor(y[test_index])
        
        # Train the best model
        print(f"Training ANN on Dpar (outer fold {k_of + 1} of {K_FOLDS}).")
        net, final_loss, learning_curve = train_neural_net(best_model,
                                                           loss_fn,
                                                           X=X_train,
                                                           y=y_train,
                                                           n_replicates=1,
                                                           max_iter=10000 if not TEST else 100)

        # Calculate the new errors for the best model
        
        # Determine estimated value    
        y_test_est = net(X_test)
        
        # Determine errors and mse
        se = (y_test_est.float()-y_test.float())**2                     # squared error
        mse = (sum(se).type(torch.float)/len(y_test)).data.numpy()      # mean
        mse = float(mse)
        # Save results
        CV_table_results["ANN"].append(np.array([k_of, best_hu, mse]))

        
np.set_printoptions(suppress=True, linewidth=np.inf)

# write files 
CV_table_results_array = CV_table_results["ANN"]
write_outfile(CV_table_results_array,
              name="2-level_CV_table",
              header="outer_fold, ANN_best_hu, ANN_error")

all_result_array = np.vstack(tuple(all_data["ANN"]))
write_outfile(all_result_array,
              name="2lCV_all_data",
              header="model_index, k_of, k_if, best_hu, ANN_error, Nval, ANN_train_error")



























