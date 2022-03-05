import matplotlib.pyplot as plt
import numpy as np
import pickle

result_path = "result_alpha/"
plot_path = "plots_osmd_alpha/"
std_plot = 1.0

alpha_list = [1e-2, 0.1, 0.4, 0.7, 0.9, 1.0] 
color_list = ['g', 'orange', 'c', 'm', 'y', 'gray']

# visualiztion
# plot of both loss and regret
fig, axes = plt.subplots(2, 3, figsize=[24., 12.])
    
sigma_list = [1.0, 3.0, 10.0] # parameter controlling the level of heterogeneity
for j, sigma in enumerate(sigma_list):
    # load the result
    with open(result_path + "log_loss_list_optimal_mean_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_loss_list_optimal_mean = pickle.load(f)

    with open(result_path + "log_loss_list_optimal_std_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_loss_list_optimal_std = pickle.load(f)

    with open(result_path + "log_loss_list_Adaptive_OSMD_mean_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_loss_list_Adaptive_OSMD_mean = pickle.load(f)

    with open(result_path + "log_loss_list_Adaptive_OSMD_std_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_loss_list_Adaptive_OSMD_std = pickle.load(f)

    # load the result
    with open(result_path + "log_regret_list_Adaptive_OSMD_mean_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_regret_list_Adaptive_OSMD_mean = pickle.load(f)

    with open(result_path + "log_regret_list_Adaptive_OSMD_std_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_regret_list_Adaptive_OSMD_std = pickle.load(f)

    n_iter = len(log_loss_list_optimal_mean)

    for i in range(len(alpha_list)):
        axes[0][j].plot(np.arange(1, n_iter+1), log_loss_list_Adaptive_OSMD_mean[i, :], color=color_list[i], label=r'$\alpha_{\star}=$'+str(alpha_list[i]), linewidth=3)
        axes[0][j].fill_between(np.arange(1, n_iter+1), log_loss_list_Adaptive_OSMD_mean[i, :] - std_plot*log_loss_list_Adaptive_OSMD_std[i, :], 
                    log_loss_list_Adaptive_OSMD_mean[i, :] + std_plot*log_loss_list_Adaptive_OSMD_std[i, :], facecolor=color_list[i], alpha=0.2,
                    edgecolor=color_list[i], linestyle='dashdot')

    axes[0][j].plot(np.arange(1, n_iter+1), log_loss_list_optimal_mean, color="b", label="Optimal", linewidth=3)
    axes[0][j].fill_between(np.arange(1, n_iter+1), log_loss_list_optimal_mean - std_plot*log_loss_list_optimal_std, 
                log_loss_list_optimal_mean + std_plot*log_loss_list_optimal_std, facecolor='b', alpha=0.2,
                edgecolor='b', linestyle='dashdot')

    axes[0][j].set_title(r'$\sigma=$'+str(sigma), fontsize=25)
    
    #axes[0][j].set_xlabel("t", fontsize=25)
    axes[0][j].tick_params(axis='x', labelsize=25)
    
    axes[0][j].tick_params(axis='y', labelsize=25)
    if j == 0:
       axes[0][j].set_ylabel("log(loss)", fontsize=25)

    box = axes[0][j].get_position()
    axes[0][j].set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])

    for i in range(len(alpha_list)):
        axes[1][j].plot(np.arange(1, n_iter+1), log_regret_list_Adaptive_OSMD_mean[i, :], color=color_list[i], label=r'$\alpha_{\star}=$'+str(alpha_list[i]), linewidth=3)
        axes[1][j].fill_between(np.arange(1, n_iter+1), log_regret_list_Adaptive_OSMD_mean[i, :] - std_plot*log_regret_list_Adaptive_OSMD_std[i, :], 
                log_regret_list_Adaptive_OSMD_mean[i, :] + std_plot*log_regret_list_Adaptive_OSMD_std[i, :], facecolor=color_list[i], alpha=0.2,
                edgecolor=color_list[i], linestyle='dashdot')
    #axes[1][j].set_title(r'$\sigma=$'+str(sigma), fontsize=25)
    
    axes[1][j].set_xlabel("t", fontsize=25)
    axes[1][j].tick_params(axis='x', labelsize=25)
    
    axes[1][j].tick_params(axis='y', labelsize=25)
    if j == 0:
       axes[1][j].set_ylabel("log(regret)", fontsize=25)

    box = axes[1][j].get_position()
    axes[1][j].set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])

labels_list = []
for alpha in alpha_list:
    labels_list.append(r'$\alpha=$'+str(alpha))
labels_list.append('Optimal')

fig.legend(labels=labels_list, loc='lower center', ncol=len(labels_list), fontsize=25)
fig.savefig(plot_path+'loss_regret_AdaOSMD_alpha.png', dpi=400, bbox_inches='tight', format='png')
plt.close(fig)

# visualiztion
# plot of loss
fig, axes = plt.subplots(1, 3, figsize=[24., 8.])
    
sigma_list = [1.0, 3.0, 10.0] # parameter controlling the level of heterogeneity
for j, sigma in enumerate(sigma_list):
    # load the result
    with open(result_path + "log_loss_list_optimal_mean_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_loss_list_optimal_mean = pickle.load(f)

    with open(result_path + "log_loss_list_optimal_std_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_loss_list_optimal_std = pickle.load(f)

    with open(result_path + "log_loss_list_Adaptive_OSMD_mean_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_loss_list_Adaptive_OSMD_mean = pickle.load(f)

    with open(result_path + "log_loss_list_Adaptive_OSMD_std_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_loss_list_Adaptive_OSMD_std = pickle.load(f)

    n_iter = len(log_loss_list_optimal_mean)

    for i in range(len(alpha_list)):
        axes[j].plot(np.arange(1, n_iter+1), log_loss_list_Adaptive_OSMD_mean[i, :], color=color_list[i], label=r'$\alpha_{\star}=$'+str(alpha_list[i]), linewidth=3)
        axes[j].fill_between(np.arange(1, n_iter+1), log_loss_list_Adaptive_OSMD_mean[i, :] - std_plot*log_loss_list_Adaptive_OSMD_std[i, :], 
                    log_loss_list_Adaptive_OSMD_mean[i, :] + std_plot*log_loss_list_Adaptive_OSMD_std[i, :], facecolor=color_list[i], alpha=0.2,
                    edgecolor=color_list[i], linestyle='dashdot')

    axes[j].plot(np.arange(1, n_iter+1), log_loss_list_optimal_mean, color="b", label="Optimal", linewidth=3)
    axes[j].fill_between(np.arange(1, n_iter+1), log_loss_list_optimal_mean - std_plot*log_loss_list_optimal_std, 
                log_loss_list_optimal_mean + std_plot*log_loss_list_optimal_std, facecolor='b', alpha=0.2,
                edgecolor='b', linestyle='dashdot')

    axes[j].set_title(r'$\sigma=$'+str(sigma), fontsize=25)
    
    axes[j].set_xlabel("t", fontsize=25)
    axes[j].tick_params(axis='x', labelsize=25)
    
    axes[j].tick_params(axis='y', labelsize=25)
    if j == 0:
       axes[j].set_ylabel("log(loss)", fontsize=25)

    box = axes[j].get_position()
    axes[j].set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])

labels_list = []
for alpha in alpha_list:
    labels_list.append(r'$\alpha=$'+str(alpha))
labels_list.append('Optimal')

fig.legend(labels=labels_list, loc='lower center', ncol=len(labels_list), fontsize=25)
fig.savefig(plot_path+'loss_AdaOSMD_alpha.png', dpi=400, bbox_inches='tight', format='png')
plt.close(fig)

# plot of regret
fig, axes = plt.subplots(1, 3, figsize=[24., 8.])
    
sigma_list = [1.0, 3.0, 10.0] # parameter controlling the level of heterogeneity
for j, sigma in enumerate(sigma_list):
    # load the result
    with open(result_path + "log_regret_list_Adaptive_OSMD_mean_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_regret_list_Adaptive_OSMD_mean = pickle.load(f)

    with open(result_path + "log_regret_list_Adaptive_OSMD_std_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_regret_list_Adaptive_OSMD_std = pickle.load(f)

    n_iter = len(log_loss_list_optimal_mean)

    for i in range(len(alpha_list)):
        axes[j].plot(np.arange(1, n_iter+1), log_regret_list_Adaptive_OSMD_mean[i, :], color=color_list[i], label=r'$\alpha_{\star}=$'+str(alpha_list[i]), linewidth=3)
        axes[j].fill_between(np.arange(1, n_iter+1), log_regret_list_Adaptive_OSMD_mean[i, :] - std_plot*log_regret_list_Adaptive_OSMD_std[i, :], 
                log_regret_list_Adaptive_OSMD_mean[i, :] + std_plot*log_regret_list_Adaptive_OSMD_std[i, :], facecolor=color_list[i], alpha=0.2,
                edgecolor=color_list[i], linestyle='dashdot')
    axes[j].set_title(r'$\sigma=$'+str(sigma), fontsize=25)
    
    axes[j].set_xlabel("t", fontsize=25)
    axes[j].tick_params(axis='x', labelsize=25)
    
    axes[j].tick_params(axis='y', labelsize=25)
    if j == 0:
       axes[j].set_ylabel("log(regret)", fontsize=25)

    box = axes[j].get_position()
    axes[j].set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])

labels_list = []
for alpha in alpha_list:
    labels_list.append(r'$\alpha=$'+str(alpha))

fig.legend(labels=labels_list, loc='lower center', ncol=len(labels_list), fontsize=25)
fig.savefig(plot_path+'regret_AdaOSMD_alpha.png', dpi=400, bbox_inches='tight', format='png')
plt.close(fig)

