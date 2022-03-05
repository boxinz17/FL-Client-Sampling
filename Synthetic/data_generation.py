import numpy as np
import pickle
from algorithms import *

M = 100  # total number of clients
n = 100  # total number of samples per client
d = 10  # dimension size of parameter
K = 5  # number of clients chosen at each round
B_bar = 10  # mini-batch size for computing local gradients

kappa = 25  # condition number of Sigma
Sigma_diag = kappa ** (np.arange(d) / (d-1) - 1)  # diagonal vector of Sigma matrix

sigma_list = [1.0, 3.0, 10.0] # parameter controlling the level of heterogeneity
result_path = "data/"  # the path to save results

simulated_data = []

for sigma in sigma_list:
    print("sigma={}".format(sigma))
    np.random.seed(111)
    w_true = np.random.normal(loc=10.0, scale=3.0, size=d)  # true parameter to generate data
    s = np.exp(np.random.normal(loc=0.0, scale=sigma, size=M))  # the vector of s_m's
    s = s / s.max() * 10.0  # rescale the maximum of s

    # Generate data
    data_list = []
    for m in range(M):
        X = np.zeros((n,d))
        for i in range(n):
            X[i,:] = np.random.normal(loc=np.zeros(d), scale=s[m]*Sigma_diag)
        y = np.matmul(X, w_true) + np.random.normal(loc=0.0, scale=0.1, size=n)
        data_list.append((X,y))

    simulated_data.append(data_list)

# Save results
with open(result_path + "simulated_data.txt", "wb") as f:
    pickle.dump(simulated_data, f)