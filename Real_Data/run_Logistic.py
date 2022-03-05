import numpy as np
import torch
import pickle

from algorithms_cv import *

from data_load import *

# Setup device
device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

# Initialize parameters
n_commun = 1000  # number of rounds
lr_list = [0.075, 0.075, 0.03]  # learning rate for sgd

M = 500  # number of clients
K = 10  # number of clients chosen at each round
n_train_list = ([1, 10, 30, 100], [0.65, 0.2, 0.1, 0.05])  # number of training samples per client
B_bar = 5  # local mini-batch size, if n_train < B_bar, then we let B_bar = n_train
n_val = 10  # number of validation samples per client
n_rep = 5  # number of repetitions

rd_seed_list = np.arange(start=100, stop=100+n_rep)  # random seed list

data_name_list = ['FMNIST']  # name of dataset

# Initialize parameters for Ada-OSMD
alpha_star = 0.4

loss_list_Ada_OSMD = np.zeros((len(data_name_list), n_rep, n_commun))
accu_list_Ada_OSMD = np.zeros((len(data_name_list), n_rep, n_commun))
loss_list_uniform = np.zeros((len(data_name_list), n_rep, n_commun))
accu_list_uniform = np.zeros((len(data_name_list), n_rep, n_commun))

train_samples_dist = np.zeros((len(data_name_list), M))

for i, data_name in enumerate(data_name_list):
    # Load the data
    print('\n')
    print("data_name: " + data_name)
    
    train_loader_list, train_list, val_loader_list, val_list = data_prepare(data_name, M, n_train_list, n_val, batch_size=64, rd_seed=111)
    print("Data has been prepared!")
    
    # record the training samples distribution
    for m in range(M):
        train_samples_dist[i, m] = len(train_list[m])
    
    with open('result_logistic/train_samples_dist_' + str(data_name) +'.pickle', 'wb') as handle:
        pickle.dump(train_samples_dist[i, :], handle)

    # set up learning rate
    lr = lr_list[i]

    for j in range(n_rep):
        # Set up random seed
        rd_seed = rd_seed_list[j]
        np.random.seed(rd_seed)

        # start training by uniform sampling
        print("Training by uniform sampling starts!")
        loss_list_uniform[i, j, :], accu_list_uniform[i, j, :] = train_uniform(train_loader_list, train_list, val_loader_list, K, B_bar, n_commun, lr, device)

        # start training by Ada-OSMD sampling
        print("Training by Ada-OSMD sampling starts!")
        loss_list_Ada_OSMD[i, j, :], accu_list_Ada_OSMD[i, j, :] = train_Ada_OSMD(train_loader_list, train_list, val_loader_list, alpha_star, K, B_bar, n_commun, lr, device)
    
        print("Repetition: {} finished!".format(j+1))
        
        # save the results
        with open('result_logistic/loss_list_uniform_' + data_name + '_rdseed=' + str(rd_seed) +'.pickle', 'wb') as handle:
            pickle.dump(loss_list_uniform[i, j, :], handle)

        with open('result_logistic/accu_list_uniform_' + data_name + '_rdseed=' + str(rd_seed) +'.pickle', 'wb') as handle:
            pickle.dump(accu_list_uniform[i, j, :], handle)

        with open('result_logistic/loss_list_Ada_OSMD_' + data_name + '_rdseed=' + str(rd_seed) +'.pickle', 'wb') as handle:
            pickle.dump(loss_list_Ada_OSMD[i, j, :], handle)

        with open('result_logistic/accu_list_Ada_OSMD_' + data_name + '_rdseed=' + str(rd_seed) +'.pickle', 'wb') as handle:
            pickle.dump(accu_list_Ada_OSMD[i, j, :], handle)

# save the results
with open('result_logistic/loss_list_Ada_OSMD.pickle', 'wb') as handle:
    pickle.dump(loss_list_Ada_OSMD, handle)

with open('result_logistic/accu_list_Ada_OSMD.pickle', 'wb') as handle:
    pickle.dump(accu_list_Ada_OSMD, handle)

with open('result_logistic/loss_list_uniform.pickle', 'wb') as handle:
    pickle.dump(loss_list_uniform, handle)

with open('result_logistic/accu_list_uniform.pickle', 'wb') as handle:
    pickle.dump(accu_list_uniform, handle)


