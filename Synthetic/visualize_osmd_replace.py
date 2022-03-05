import matplotlib.pyplot as plt
import numpy as np
import pickle

result_path = "result/"
plot_path = "plots_osmd_replace/"
std_plot = 1.0

# visualiztion
# plot of loss and regret
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

    with open(result_path + "log_loss_list_Adaptive_OSMD_Replace_mean_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_loss_list_Adaptive_OSMD_Replace_mean = pickle.load(f)

    with open(result_path + "log_loss_list_Adaptive_OSMD_Replace_std_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_loss_list_Adaptive_OSMD_Replace_std = pickle.load(f)

    # load the result
    with open(result_path + "log_regret_list_Adaptive_OSMD_mean_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_regret_list_Adaptive_OSMD_mean = pickle.load(f)

    with open(result_path + "log_regret_list_Adaptive_OSMD_std_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_regret_list_Adaptive_OSMD_std = pickle.load(f)

    with open(result_path + "log_regret_list_Adaptive_OSMD_Replace_mean_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_regret_list_Adaptive_OSMD_Replace_mean = pickle.load(f)

    with open(result_path + "log_regret_list_Adaptive_OSMD_Replace_std_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_regret_list_Adaptive_OSMD_Replace_std = pickle.load(f)

    n_iter = len(log_loss_list_Adaptive_OSMD_mean)

    axes[0][j].plot(np.arange(1, n_iter+1), log_loss_list_Adaptive_OSMD_mean, color="g", label=r"Ada-OSMD with $\alpha=0.4$", linewidth=3)
    axes[0][j].fill_between(np.arange(1, n_iter+1), log_loss_list_Adaptive_OSMD_mean - std_plot*log_loss_list_Adaptive_OSMD_std, 
                log_loss_list_Adaptive_OSMD_mean + std_plot*log_loss_list_Adaptive_OSMD_std, facecolor='g', alpha=0.2,
                edgecolor='g', linestyle='dashdot')

    axes[0][j].plot(np.arange(1, n_iter+1), log_loss_list_Adaptive_OSMD_Replace_mean, color="m", label=r"Ada-OSMD-woReplace with $\alpha=0.4$", linewidth=3)
    axes[0][j].fill_between(np.arange(1, n_iter+1), log_loss_list_Adaptive_OSMD_Replace_mean - std_plot*log_loss_list_Adaptive_OSMD_Replace_std, 
                log_loss_list_Adaptive_OSMD_Replace_mean + std_plot*log_loss_list_Adaptive_OSMD_Replace_std, facecolor='m', alpha=0.2,
                edgecolor='m', linestyle='dashdot')

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

    axes[1][j].plot(np.arange(1, n_iter+1), log_regret_list_Adaptive_OSMD_mean, color="g", label=r"Ada-OSMD with $\alpha=0.4$", linewidth=3)
    axes[1][j].fill_between(np.arange(1, n_iter+1), log_regret_list_Adaptive_OSMD_mean - std_plot*log_regret_list_Adaptive_OSMD_std, 
                log_regret_list_Adaptive_OSMD_mean + std_plot*log_regret_list_Adaptive_OSMD_std, facecolor='g', alpha=0.2,
                edgecolor='g', linestyle='dashdot')

    axes[1][j].plot(np.arange(1, n_iter+1), log_regret_list_Adaptive_OSMD_Replace_mean, color="m", label=r"Ada-OSMD-woReplace with $\alpha=0.4$", linewidth=3)
    axes[1][j].fill_between(np.arange(1, n_iter+1), log_regret_list_Adaptive_OSMD_Replace_mean - std_plot*log_regret_list_Adaptive_OSMD_Replace_std, 
                log_regret_list_Adaptive_OSMD_Replace_mean + std_plot*log_regret_list_Adaptive_OSMD_Replace_std, facecolor='m', alpha=0.2,
                edgecolor='m', linestyle='dashdot')

    #axes[1][j].set_title(r'$\sigma=$'+str(sigma), fontsize=25)
    
    axes[1][j].set_xlabel("t", fontsize=25)
    axes[1][j].tick_params(axis='x', labelsize=25)
    
    axes[1][j].tick_params(axis='y', labelsize=25)
    if j == 0:
       axes[1][j].set_ylabel("log(regret)", fontsize=25)

    box = axes[1][j].get_position()
    axes[1][j].set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])

fig.legend(labels=[r'Ada-OSMD with $\alpha=0.4$', r"Ada-OSMD-woReplace with $\alpha=0.4$", 'Optimal'], loc='lower center', ncol=3, fontsize=25)
fig.savefig(plot_path+'loss_regret_AdaOSMD-Replace.png', dpi=400, bbox_inches='tight')
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

    with open(result_path + "log_loss_list_Adaptive_OSMD_Replace_mean_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_loss_list_Adaptive_OSMD_Replace_mean = pickle.load(f)

    with open(result_path + "log_loss_list_Adaptive_OSMD_Replace_std_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_loss_list_Adaptive_OSMD_Replace_std = pickle.load(f)

    n_iter = len(log_loss_list_Adaptive_OSMD_mean)

    axes[j].plot(np.arange(1, n_iter+1), log_loss_list_Adaptive_OSMD_mean, color="g", label=r"Ada-OSMD with $\alpha=0.4$", linewidth=3)
    axes[j].fill_between(np.arange(1, n_iter+1), log_loss_list_Adaptive_OSMD_mean - std_plot*log_loss_list_Adaptive_OSMD_std, 
                log_loss_list_Adaptive_OSMD_mean + std_plot*log_loss_list_Adaptive_OSMD_std, facecolor='g', alpha=0.2,
                edgecolor='g', linestyle='dashdot')

    axes[j].plot(np.arange(1, n_iter+1), log_loss_list_Adaptive_OSMD_Replace_mean, color="m", label=r"Ada-OSMD-woReplace with $\alpha=0.4$", linewidth=3)
    axes[j].fill_between(np.arange(1, n_iter+1), log_loss_list_Adaptive_OSMD_Replace_mean - std_plot*log_loss_list_Adaptive_OSMD_Replace_std, 
                log_loss_list_Adaptive_OSMD_Replace_mean + std_plot*log_loss_list_Adaptive_OSMD_Replace_std, facecolor='m', alpha=0.2,
                edgecolor='m', linestyle='dashdot')

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

fig.legend(labels=[r'Ada-OSMD with $\alpha=0.4$', r"Ada-OSMD-woReplace with $\alpha=0.4$", 'Optimal'], loc='lower center', ncol=3, fontsize=25)
fig.savefig(plot_path+'loss_AdaOSMD-Replace.png', dpi=400, bbox_inches='tight')
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

    with open(result_path + "log_regret_list_Adaptive_OSMD_Replace_mean_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_regret_list_Adaptive_OSMD_Replace_mean = pickle.load(f)

    with open(result_path + "log_regret_list_Adaptive_OSMD_Replace_std_sigma=" + str(sigma) + ".txt", "rb") as f:
        log_regret_list_Adaptive_OSMD_Replace_std = pickle.load(f)

    n_iter = len(log_regret_list_Adaptive_OSMD_mean)

    axes[j].plot(np.arange(1, n_iter+1), log_regret_list_Adaptive_OSMD_mean, color="g", label=r"Ada-OSMD with $\alpha=0.4$", linewidth=3)
    axes[j].fill_between(np.arange(1, n_iter+1), log_regret_list_Adaptive_OSMD_mean - std_plot*log_regret_list_Adaptive_OSMD_std, 
                log_regret_list_Adaptive_OSMD_mean + std_plot*log_regret_list_Adaptive_OSMD_std, facecolor='g', alpha=0.2,
                edgecolor='g', linestyle='dashdot')

    axes[j].plot(np.arange(1, n_iter+1), log_regret_list_Adaptive_OSMD_Replace_mean, color="m", label=r"Ada-OSMD-woReplace with $\alpha=0.4$", linewidth=3)
    axes[j].fill_between(np.arange(1, n_iter+1), log_regret_list_Adaptive_OSMD_Replace_mean - std_plot*log_regret_list_Adaptive_OSMD_Replace_std, 
                log_regret_list_Adaptive_OSMD_Replace_mean + std_plot*log_regret_list_Adaptive_OSMD_Replace_std, facecolor='m', alpha=0.2,
                edgecolor='m', linestyle='dashdot')

    axes[j].set_title(r'$\sigma=$'+str(sigma), fontsize=25)
    
    axes[j].set_xlabel("t", fontsize=25)
    axes[j].tick_params(axis='x', labelsize=25)
    
    axes[j].tick_params(axis='y', labelsize=25)
    if j == 0:
       axes[j].set_ylabel("log(regret)", fontsize=25)

    box = axes[j].get_position()
    axes[j].set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])

fig.legend(labels=[r'Ada-OSMD with $\alpha=0.4$', r"Ada-OSMD-woReplace with $\alpha=0.4$"], loc='lower center', ncol=2, fontsize=25)
fig.savefig(plot_path+'regret_AdaOSMD-Replace.png', dpi=400, bbox_inches='tight')
plt.close(fig)

