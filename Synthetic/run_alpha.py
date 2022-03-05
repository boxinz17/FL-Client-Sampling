import numpy as np
import pickle
from algorithms import *

M = 100  # total number of clients
n = 100  # total number of samples per client
d = 10  # dimension size of parameter
K = 5  # number of clients chosen at each round
B_bar = 10  # mini-batch size for computing local gradients

# list of alpha_star we want to try
alpha_list = [1e-2, 0.1, 0.4, 0.7, 0.9, 1.0] 

# Perform SGD
n_iter = 1000  # number of iterations
eta_SGD = 0.1  # learning rate for SGD
n_repeat = 10  # number of times for repeating the experiment

sigma_list = [1.0, 3.0, 10.0] # parameter controlling the level of heterogeneity
result_path = "result_alpha/" # the path to save results
data_path = "data/"  # the path to load the data

# load the data
with open(data_path + "simulated_data.txt", "rb") as f:
    simulated_data = pickle.load(f)

for i_data, sigma in enumerate(sigma_list):
    print("sigma={}".format(sigma))
    np.random.seed(111)

    data_list = simulated_data[i_data]

    # Parameter for Adaptive OSMD-sampler
    alpha_star = 0.4
    bar_a1 = bar_a1_est(B_bar, data_list)

    loss_list_optimal = np.zeros((n_repeat, n_iter))
    loss_list_Adaptive_OSMD = np.zeros((len(alpha_list), n_repeat, n_iter))

    regret_list_Adaptive_OSMD = np.zeros((len(alpha_list), n_repeat, n_iter))
    
    for i, alpha_star in enumerate(alpha_list):
        for k_rep in range(n_repeat):
            loss_list_Adaptive_OSMD[i, k_rep, :], regret_list_Adaptive_OSMD[i, k_rep, :] = train_Ada_OSMD(K, B_bar, data_list, eta_SGD, alpha_star, n_iter)
            print("Ada-OSMD with alpha={} Repeat: {} finished!".format(alpha_star, k_rep+1))

    for k_rep in range(n_repeat):
        loss_list_optimal[k_rep, :] = train_optimal(K, B_bar, data_list, n_iter, eta_SGD)
        print("Optimal Repeat: {} finished!".format(k_rep+1))

    # Compute loss mean and std
    log_loss_list_optimal_mean = np.log(loss_list_optimal).mean(0)
    log_loss_list_optimal_std = np.log(loss_list_optimal).std(0)

    log_loss_list_Adaptive_OSMD_mean = np.log(loss_list_Adaptive_OSMD).mean(1)
    log_loss_list_Adaptive_OSMD_std = np.log(loss_list_Adaptive_OSMD).std(1)

    # Compute regret mean and std
    log_regret_list_Adaptive_OSMD_mean = np.log(regret_list_Adaptive_OSMD.cumsum(2)).mean(1)
    log_regret_list_Adaptive_OSMD_std = np.log(regret_list_Adaptive_OSMD.cumsum(2)).std(1)

    # Save results
    with open(result_path + "log_loss_list_optimal_mean_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_loss_list_optimal_mean, f)

    with open(result_path + "log_loss_list_optimal_std_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_loss_list_optimal_std, f)

    with open(result_path + "log_loss_list_Adaptive_OSMD_mean_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_loss_list_Adaptive_OSMD_mean, f)

    with open(result_path + "log_loss_list_Adaptive_OSMD_std_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_loss_list_Adaptive_OSMD_std, f)

    with open(result_path + "log_regret_list_Adaptive_OSMD_mean_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_regret_list_Adaptive_OSMD_mean, f)

    with open(result_path + "log_regret_list_Adaptive_OSMD_std_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_regret_list_Adaptive_OSMD_std, f)