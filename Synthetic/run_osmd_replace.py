import numpy as np
import pickle
from algorithms import *

M = 100  # total number of clients
n = 100  # total number of samples per client
d = 10  # dimension size of parameter
K = 5  # number of clients chosen at each round
B_bar = 10  # mini-batch size for computing local gradients

# Perform SGD
n_iter = 1000  # number of iterations
eta_SGD = 0.1  # learning rate for SGD
n_repeat = 10  # number of times for repeating the experiment

sigma_list = [1.0, 3.0, 10.0] # parameter controlling the level of heterogeneity
result_path = "result/" # the path to save results
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

    loss_list_Adaptive_OSMD_Replace = np.zeros((n_repeat, n_iter))
    regret_list_Adaptive_OSMD_Replace = np.zeros((n_repeat, n_iter))
    for k_rep in range(n_repeat):
        loss_list_Adaptive_OSMD_Replace[k_rep, :], regret_list_Adaptive_OSMD_Replace[k_rep, :] = train_Ada_OSMD_woreplace(K, B_bar, data_list, eta_SGD, alpha_star, n_iter)
        print("Repeat: {} finished!".format(k_rep+1))

    # Compute loss mean and std
    log_loss_list_Adaptive_OSMD_Replace_mean = np.log(loss_list_Adaptive_OSMD_Replace).mean(0)
    log_loss_list_Adaptive_OSMD_Replace_std = np.log(loss_list_Adaptive_OSMD_Replace).std(0)

    # Compute regret mean and std
    log_regret_list_Adaptive_OSMD_Replace_mean = np.log(regret_list_Adaptive_OSMD_Replace.cumsum(1)).mean(0)
    log_regret_list_Adaptive_OSMD_Replace_std = np.log(regret_list_Adaptive_OSMD_Replace.cumsum(1)).std(0)

    # Save results
    with open(result_path + "log_loss_list_Adaptive_OSMD_Replace_mean_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_loss_list_Adaptive_OSMD_Replace_mean, f)

    with open(result_path + "log_loss_list_Adaptive_OSMD_Replace_std_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_loss_list_Adaptive_OSMD_Replace_std, f)

    with open(result_path + "log_regret_list_Adaptive_OSMD_Replace_mean_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_regret_list_Adaptive_OSMD_Replace_mean, f)

    with open(result_path + "log_regret_list_Adaptive_OSMD_Replace_std_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_regret_list_Adaptive_OSMD_Replace_std, f)