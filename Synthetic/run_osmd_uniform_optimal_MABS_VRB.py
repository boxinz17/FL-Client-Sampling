import numpy as np
import pickle
from algorithms import *

M = 100  # total number of clients
n = 100  # total number of samples per client
d = 10  # dimension size of parameter
K = 5  # number of clients chosen at each round
B_bar = 10  # mini-batch size for computing local gradients

# Parameter for MABS
alpha_MABS = 0.4

# Parameter for VRB
alpha_VRB = 0.4

# Perform SGD
n_iter = 1000  # number of iterations
eta_SGD = 0.1  # learning rate for SGD
n_repeat = 10  # number of times for repeating the experiment

sigma_list = [1.0, 3.0, 10.0] # parameter controlling the level of heterogeneity
result_path = "result/"  # the path to save results
data_path = "data/"  # the path to load the data

# load the data
with open(data_path + "simulated_data.txt", "rb") as f:
    simulated_data = pickle.load(f)

for i_data, sigma in enumerate(sigma_list):
    print("sigma={}".format(sigma))
    np.random.seed(111)
    
    data_list = simulated_data[i_data]

    # Parameter for Adaptive OSMD-sampler
    alpha_OSMD = 0.4

    loss_list_uniform = np.zeros((n_repeat, n_iter))
    loss_list_optimal = np.zeros((n_repeat, n_iter))
    loss_list_Adaptive_OSMD = np.zeros((n_repeat, n_iter))
    loss_list_MABS = np.zeros((n_repeat, n_iter))
    loss_list_VRB = np.zeros((n_repeat, n_iter))

    regret_list_uniform = np.zeros((n_repeat, n_iter))
    regret_list_Adaptive_OSMD = np.zeros((n_repeat, n_iter))
    regret_list_MABS = np.zeros((n_repeat, n_iter))
    regret_list_VRB = np.zeros((n_repeat, n_iter))

    for k_rep in range(n_repeat):
        loss_list_uniform[k_rep, :], regret_list_uniform[k_rep, :] = train_uniform(K, B_bar, data_list, n_iter, eta_SGD)
        loss_list_optimal[k_rep, :] = train_optimal(K, B_bar, data_list, n_iter, eta_SGD)
        loss_list_Adaptive_OSMD[k_rep, :], regret_list_Adaptive_OSMD[k_rep, :] = train_Ada_OSMD(K, B_bar, data_list, eta_SGD, alpha_OSMD, n_iter)
        loss_list_MABS[k_rep, :], regret_list_MABS[k_rep, :] = train_MABS(K, B_bar, data_list, n_iter, eta_SGD, alpha_MABS)
        loss_list_VRB[k_rep, :], regret_list_VRB[k_rep, :] = train_VRB(K, B_bar, data_list, n_iter, eta_SGD, alpha_VRB)
        print("Repeat: {} finished!".format(k_rep+1))

    # Compute loss mean and std
    log_loss_list_uniform_mean = np.log(loss_list_uniform).mean(0)
    log_loss_list_uniform_std = np.log(loss_list_uniform).std(0)

    log_loss_list_optimal_mean = np.log(loss_list_optimal).mean(0)
    log_loss_list_optimal_std = np.log(loss_list_optimal).std(0)

    log_loss_list_Adaptive_OSMD_mean = np.log(loss_list_Adaptive_OSMD).mean(0)
    log_loss_list_Adaptive_OSMD_std = np.log(loss_list_Adaptive_OSMD).std(0)

    log_loss_list_MABS_mean = np.log(loss_list_MABS).mean(0)
    log_loss_list_MABS_std = np.log(loss_list_MABS).std(0)

    log_loss_list_VRB_mean = np.log(loss_list_VRB).mean(0)
    log_loss_list_VRB_std = np.log(loss_list_VRB).std(0)

    # Compute regret mean and std
    log_regret_list_uniform_mean = np.log(regret_list_uniform.cumsum(1)).mean(0)
    log_regret_list_uniform_std = np.log(regret_list_uniform.cumsum(1)).std(0)

    log_regret_list_Adaptive_OSMD_mean = np.log(regret_list_Adaptive_OSMD.cumsum(1)).mean(0)
    log_regret_list_Adaptive_OSMD_std = np.log(regret_list_Adaptive_OSMD.cumsum(1)).std(0)

    log_regret_list_MABS_mean = np.log(regret_list_MABS.cumsum(1)).mean(0)
    log_regret_list_MABS_std = np.log(regret_list_MABS.cumsum(1)).std(0)

    log_regret_list_VRB_mean = np.log(regret_list_VRB.cumsum(1)).mean(0)
    log_regret_list_VRB_std = np.log(regret_list_VRB.cumsum(1)).std(0)

    # Save results
    with open(result_path + "log_loss_list_uniform_mean_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_loss_list_uniform_mean, f)

    with open(result_path +  "log_loss_list_uniform_std_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_loss_list_uniform_std, f)

    with open(result_path + "log_loss_list_optimal_mean_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_loss_list_optimal_mean, f)

    with open(result_path + "log_loss_list_optimal_std_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_loss_list_optimal_std, f)

    with open(result_path + "log_loss_list_Adaptive_OSMD_mean_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_loss_list_Adaptive_OSMD_mean, f)

    with open(result_path + "log_loss_list_Adaptive_OSMD_std_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_loss_list_Adaptive_OSMD_std, f)

    with open(result_path + "log_loss_list_MABS_mean_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_loss_list_MABS_mean, f)

    with open(result_path + "log_loss_list_MABS_std_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_loss_list_MABS_std, f)

    with open(result_path + "log_loss_list_VRB_mean_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_loss_list_VRB_mean, f)

    with open(result_path + "log_loss_list_VRB_std_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_loss_list_VRB_std, f)

    with open(result_path + "log_regret_list_uniform_mean_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_regret_list_uniform_mean, f)

    with open(result_path + "log_regret_list_uniform_std_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_regret_list_uniform_std, f)

    with open(result_path + "log_regret_list_Adaptive_OSMD_mean_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_regret_list_Adaptive_OSMD_mean, f)

    with open(result_path + "log_regret_list_Adaptive_OSMD_std_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_regret_list_Adaptive_OSMD_std, f)

    with open(result_path + "log_regret_list_MABS_mean_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_regret_list_MABS_mean, f)

    with open(result_path + "log_regret_list_MABS_std_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_regret_list_MABS_std, f)

    with open(result_path + "log_regret_list_VRB_mean_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_regret_list_VRB_mean, f)

    with open(result_path + "log_regret_list_VRB_std_sigma=" + str(sigma) + ".txt", "wb") as f:
        pickle.dump(log_regret_list_VRB_std, f)
