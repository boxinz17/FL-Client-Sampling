import pickle
import numpy as np
import matplotlib
from matplotlib.ticker import FormatStrFormatter
from matplotlib import pyplot as plt
matplotlib.rcParams['text.usetex'] = True

data_name_list = ['MNIST', 'KMNIST', 'FMNIST']  # name of dataset
n_rep = 5  # number of repetitions
rd_seed_list = np.arange(start=100, stop=100+n_rep)  # random seed list
n_commun = 1000  # number of rounds
M = 500 # number of clients

loss_list_Ada_OSMD = np.zeros((len(data_name_list), n_rep, n_commun))
accu_list_Ada_OSMD = np.zeros((len(data_name_list), n_rep, n_commun))
loss_list_uniform = np.zeros((len(data_name_list), n_rep, n_commun))
accu_list_uniform = np.zeros((len(data_name_list), n_rep, n_commun))

# load samples distribution info
train_samples_dist = np.zeros((len(data_name_list), M))
for i, data_name in enumerate(data_name_list):
    with open('result_logistic/train_samples_dist_' + str(data_name) +'.pickle', 'rb') as handle:
        train_samples_dist[i, :] = pickle.load(handle)

# load the results
for i, data_name in enumerate(data_name_list):
    for j in range(n_rep):
        rd_seed = rd_seed_list[j]
        with open('result_logistic/loss_list_uniform_' + data_name + '_rdseed=' + str(rd_seed) +'.pickle', 'rb') as handle:
            loss_list_uniform[i, j, :] = pickle.load(handle)

        with open('result_logistic/accu_list_uniform_' + data_name + '_rdseed=' + str(rd_seed) +'.pickle', 'rb') as handle:
            accu_list_uniform[i, j, :] = pickle.load(handle)
        
        with open('result_logistic/loss_list_Ada_OSMD_' + data_name + '_rdseed=' + str(rd_seed) +'.pickle', 'rb') as handle:
            loss_list_Ada_OSMD[i, j, :] = pickle.load(handle)

        with open('result_logistic/accu_list_Ada_OSMD_' + data_name + '_rdseed=' + str(rd_seed) +'.pickle', 'rb') as handle:
            accu_list_Ada_OSMD[i, j, :] = pickle.load(handle)

loss_list_uniform_logmean = np.log(loss_list_uniform).mean(1)
accu_list_uniform = accu_list_uniform.mean(1)
loss_list_Ada_OSMD_logmean = np.log(loss_list_Ada_OSMD).mean(1)
accu_list_Ada_OSMD = accu_list_Ada_OSMD.mean(1)

# plot
std_plot = 1.0

# plot samples distribution info
fig, ax = plt.subplots(1, 1, figsize=[20., 8.])
ax.hist(train_samples_dist[0, :], bins=100, density=False)
ax.set_xlabel('number of samples', fontsize=40)
ax.set_ylabel('number of users', fontsize=40)
ax.set_title('Sample size distribution', fontsize=40)
ax.tick_params(axis='both', which='major', labelsize=40)
plt.savefig('plots_logistic/samples_distribution.png', dpi=400, bbox_inches='tight')
plt.close(fig)

# plot loss
fig, axes = plt.subplots(1, 3, figsize=[32., 12.])
for i, data_name in enumerate(data_name_list):
    axes[i].plot(np.arange(1, n_commun+1), loss_list_uniform_logmean[i, :], color="r", label="Uniform", linewidth=3)
    axes[i].plot(np.arange(1, n_commun+1), loss_list_Ada_OSMD_logmean[i, :], color="g", label=r"Ada-OSMD with $\alpha=0.4$")
    axes[i].tick_params(axis='both', which='major', labelsize=25)
    axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[i].set_ylabel(r'log(Loss)', fontsize=25)
    axes[i].set_xlabel('Communication Rounds', fontsize=25)
    axes[i].set_title(data_name, fontsize=25)
fig.legend(labels=['Uniform', r"Ada-OSMD with $\alpha=0.4$"], loc='upper center', ncol=2, fontsize=25)
plt.savefig('plots_logistic/logistic_osmd_uniform_loss_cv.png', dpi=400, bbox_inches='tight')
plt.savefig('plots_logistic/logistic_osmd_uniform_loss_cv.eps', dpi=400, bbox_inches='tight', format='eps')
plt.close(fig)

# plot accuracy
fig, axes = plt.subplots(1, 3, figsize=[32., 12.])
for i, data_name in enumerate(data_name_list):
    axes[i].plot(np.arange(1, n_commun+1), accu_list_uniform[i, :], color="r", label="Uniform", linewidth=3)
    axes[i].plot(np.arange(1, n_commun+1), accu_list_Ada_OSMD[i, :], color="g", label=r"Ada-OSMD with $\alpha=0.4$")
    axes[i].tick_params(axis='both', which='major', labelsize=25)
    axes[i].yaxis.set_ticks(np.linspace(0.0, 1.0, num=5))
    axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[i].set_ylabel(r'Accuracy', fontsize=25)
    axes[i].set_xlabel('Communication Rounds', fontsize=25)
    axes[i].set_title(data_name, fontsize=25)
fig.legend(labels=['Uniform', r"Ada-OSMD with $\alpha=0.4$"], loc='upper center', ncol=2, fontsize=25)
plt.savefig('plots_logistic/logistic_osmd_uniform_accuracy_cv.png', dpi=400, bbox_inches='tight')
plt.savefig('plots_logistic/logistic_osmd_uniform_accuracy_cv.eps', dpi=400, bbox_inches='tight', format='eps')
plt.close(fig)

# plot loss and accuracy
fig, axes = plt.subplots(2, 3, figsize=[32., 24.])
for i, data_name in enumerate(data_name_list):
    axes[0, i].plot(np.arange(1, n_commun+1), loss_list_uniform_logmean[i, :], color="r", label="Uniform", linewidth=3)
    axes[0, i].plot(np.arange(1, n_commun+1), loss_list_Ada_OSMD_logmean[i, :], color="g", label=r"Ada-OSMD with $\alpha=0.4$")
    axes[0, i].tick_params(axis='both', which='major', labelsize=30)
    axes[0, i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[0, i].set_ylabel(r'log(Loss)', fontsize=30)
    #axes[0, i].set_xlabel('Communication Rounds', fontsize=35)
    axes[0, i].set_title(data_name, fontsize=35)

    axes[1, i].plot(np.arange(1, n_commun+1), accu_list_uniform[i, :], color="r", label="Uniform", linewidth=3)
    axes[1, i].plot(np.arange(1, n_commun+1), accu_list_Ada_OSMD[i, :], color="g", label=r"Ada-OSMD with $\alpha=0.4$")
    axes[1, i].tick_params(axis='both', which='major', labelsize=30)
    axes[1, i].yaxis.set_ticks(np.linspace(0.0, 1.0, num=5))
    axes[1, i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[1, i].set_ylabel(r'Accuracy', fontsize=30)
    axes[1, i].set_xlabel('Communication Rounds', fontsize=35)
    #axes[1, i].set_title(data_name, fontsize=35)
fig.legend(labels=['Uniform', r"Ada-OSMD with $\alpha=0.4$"], loc='lower center', ncol=2, fontsize=35)
plt.savefig('plots_logistic/logistic_osmd_uniform_cv.png', dpi=400, bbox_inches='tight')
plt.savefig('plots_logistic/logistic_osmd_uniform_cv.eps', dpi=400, bbox_inches='tight', format='eps')
plt.close(fig)