import numpy as np

def loss_fun(w, data_list):
    M = len(data_list)
    loss = 0.0
    for m in range(M):
        X, y = data_list[m]
        loss += np.linalg.norm(y - np.matmul(X, w))**2 / (2*len(y))
    loss /= M
    return loss

def uniform_dist(M):
    # M: Number of clients
    return np.ones(M) / M

def optimal_dist(g_mat):
    # g_mat: dxM, the matrix of gradients
    M = g_mat.shape[1]
    g_norm_v = np.zeros(M)
    for m in range(M):
        g_norm_v[m] = np.linalg.norm(g_mat[:,m])
    # Set threshold for numerical stability
    prob = g_norm_v / g_norm_v.sum()
    for m in range(M):
        if prob[m] <= 1e-10:
            prob[m] = 0.0
    prob = prob / prob.sum()

    return prob


def train_uniform(K, B_bar, data_list, n_iter, eta_SGD):
    M = len(data_list)
    d = data_list[0][0].shape[1]
    n = len(data_list[0][1])
    
    loss_list = []  # list of loss values
    regret_list = []  # list of regret
    w = np.zeros(d)  # initialize w
    for t in range(n_iter):
        # set the sampling distribution for choosing clients
        prob_sampling =  uniform_dist(M)

        # Decide the batches for all clients
        batches = []
        for m in range(M):
            batch = list(np.random.choice(n, size=B_bar, replace=False))
            batches.append(batch)
        
        # choose client and make update
        G = np.zeros(d)  # sum of debiased gradients
        client_choose = np.random.choice(M, size=K, p=prob_sampling)  # choose client set with sampling distribution
        for k in range(K):
            X, y = data_list[client_choose[k]]
            batch = list(np.random.choice(n, size=B_bar, replace=False))
            X_choose = X[batch,:]
            y_choose = y[batch]
            g = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w))  # gradient of mini-batch
            g /= prob_sampling[client_choose[k]]  # debias the gradient
            G += g
        w -=  (eta_SGD / (M*K*B_bar)) * G
        loss_list.append(loss_fun(w, data_list))

        # Compute regret
        g_mat = np.zeros((d,M))
        for m in range(M):
            X, y = data_list[m]
            X_choose = X[batches[m],:]
            y_choose = y[batches[m]]
            g_mat[:,m] = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w)) / B_bar
        g_norm_v = np.zeros(M)
        for m in range(M):
            g_norm_v[m] = np.linalg.norm(g_mat[:,m])
        regret_list.append( (((1/M)*g_norm_v) ** 2 / prob_sampling).sum() - ((1/M)*g_norm_v).sum()**2 )

    return loss_list, regret_list

def train_optimal(K, B_bar, data_list, n_iter, eta_SGD):
    M = len(data_list)
    d = data_list[0][0].shape[1]
    n = len(data_list[0][1])
    
    loss_list = []  # list of loss values
    w = np.zeros(d)  # initialize w
    for t in range(n_iter):
        # Decide the batches for all clients
        batches = []
        for m in range(M):
            batch = list(np.random.choice(n, size=B_bar, replace=False))
            batches.append(batch)
        
        # Compute all mini-batch gradients
        g_mat = np.zeros((d,M))
        for m in range(M):
            X, y = data_list[m]
            X_choose = X[batches[m],:]
            y_choose = y[batches[m]]
            g_mat[:,m] = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w)) / B_bar
        # Compute the optimal client sampling distribution
        prob_sampling = optimal_dist(g_mat)
        
        # choose client and make update
        G = np.zeros(d)  # sum of debiased gradients
        client_choose = np.random.choice(M, size=K, p=prob_sampling)  # choose client set with sampling distribution
        for k in range(K):
            X, y = data_list[client_choose[k]]
            X_choose = X[batches[client_choose[k]],:]
            y_choose = y[batches[client_choose[k]]]
            g = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w))  # gradient of mini-batch
            g /= prob_sampling[client_choose[k]]  # debias the gradient
            G += g
        w -=  (eta_SGD / (M*K*B_bar)) * G
        loss_list.append(loss_fun(w, data_list))

    return loss_list

def OMD_solver(p, client_choose, a_choose, eta, K, alpha):
    M = len(p)
    
    p_new = p.copy()
    for k in range(K):
        p_new[client_choose[k]] = p[client_choose[k]] * np.exp( eta * a_choose[k] / ( (K**2)*(p[client_choose[k]]**3) ) ) 
    
    p_new_sorted = np.sort(p_new)

    m_star = 0
    for m in range(M, 0, -1):
        if p_new_sorted[m-1] * (1 - alpha * (m-1) / M) <= (alpha/M) * p_new_sorted[m-1:].sum():
            m_star = m+1
            break
    
    ss = p_new_sorted[m_star-1:].sum()
    p_new_argsort = np.argsort(p_new)
    p_hat = np.zeros(M)
    for m in range(M):
        if m+1 < m_star:
            p_hat[p_new_argsort[m]] = alpha / M
        else:
            p_hat[p_new_argsort[m]] = p_new[p_new_argsort[m]] * (1 - alpha * (m_star-1) / M) / ss
    
    return p_hat

def train_OSMD(K, B_bar, data_list, n_iter, eta_SGD, eta, alpha):
    M = len(data_list)
    d = data_list[0][0].shape[1]
    n = len(data_list[0][1])
    
    loss_list = []  # list of loss values
    regret_list = []  # list of regret
    w = np.zeros(d)  # initialize w
    # initialize the sampling distribution as uniform distribution
    prob_sampling = np.ones(M) / M
    for t in range(n_iter):
        # Decide the batches for all clients
        batches = []
        for m in range(M):
            batch = list(np.random.choice(n, size=B_bar, replace=False))
            batches.append(batch)
        
        # choose client and make update
        G = np.zeros(d)  # sum of debiased gradients
        prob_sampling /= prob_sampling.sum()  # normalize prob_sampling due to numerical error
        client_choose = np.random.choice(M, size=K, p=prob_sampling)  # choose client set with sampling distribution
        a_choose = np.zeros(K)  # feedback from environment
        for k in range(K):
            X, y = data_list[client_choose[k]]
            X_choose = X[batches[client_choose[k]],:]
            y_choose = y[batches[client_choose[k]]]
            g = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w))  # gradient of mini-batch
            a_choose[k] = ((1/M) * np.linalg.norm(g))**2
            g /= prob_sampling[client_choose[k]]  # debias the gradient
            G += g
        w -=  (eta_SGD / (M*K*B_bar)) * G  # make updates of parameter
        loss_list.append(loss_fun(w, data_list))

        # Compute regret
        g_mat = np.zeros((d,M))
        for m in range(M):
            X, y = data_list[m]
            X_choose = X[batches[m],:]
            y_choose = y[batches[m]]
            g_mat[:,m] = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w)) / B_bar
        g_norm_v = np.zeros(M)
        for m in range(M):
            g_norm_v[m] = np.linalg.norm(g_mat[:,m])
        regret_list.append( (((1/M)*g_norm_v) ** 2 / prob_sampling).sum() - ((1/M)*g_norm_v).sum()**2 )

        # make updates of sampling distribution
        prob_sampling = OMD_solver(prob_sampling, client_choose, a_choose, eta, K, alpha)

    return loss_list, regret_list

def bar_a1_est(B_bar, data_list):
    M = len(data_list)
    d = data_list[0][0].shape[1]
    n = len(data_list[0][1])

    w = np.zeros(d)  # initialize w
    batches = []
    for m in range(M):
        batch = list(np.random.choice(n, size=B_bar, replace=False))
        batches.append(batch)

    g_mat = np.zeros((d,M))
    for m in range(M):
        X, y = data_list[m]
        X_choose = X[batches[m],:]
        y_choose = y[batches[m]]
        g_mat[:,m] = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w)) / B_bar

    g_norm_v = np.zeros(M)
    for m in range(M):
        g_norm_v[m] = np.linalg.norm(g_mat[:,m])

    return ((1/M)*g_norm_v.max())**2

def lr_set_fun(K, alpha, M, bar_a1, n_iter):
    E = np.ceil( 0.5 * np.log2( (np.log(M)+4*(n_iter-1)*np.log(M/alpha)) / np.log(M) ) ) + 1
    E = int(E)
    lr_set = []
    for e in range(E):
        lr_set.append( 2**e * (K**2 * alpha**3 / (M**3 * bar_a1)) * np.sqrt(np.log(M)/(2*n_iter)) )
    return lr_set

def Adaptive_OSMD_Expert(p, client_choose, grad, eta, K, alpha):
    M = len(p)
    
    p_new = p.copy()
    for k in range(K):
        p_new[client_choose[k]] = p[client_choose[k]] * np.exp( - eta * grad[client_choose[k]] ) 
    
    p_new_sorted = np.sort(p_new)

    m_star = 0
    for m in range(M, 0, -1):
        if p_new_sorted[m-1] * (1 - alpha * (m-1) / M) <= (alpha/M) * p_new_sorted[m-1:].sum():
            m_star = m+1
            break
    
    ss = p_new_sorted[m_star-1:].sum()
    p_new_argsort = np.argsort(p_new)
    p_hat = np.zeros(M)
    for m in range(M):
        if m+1 < m_star:
            p_hat[p_new_argsort[m]] = alpha / M
        else:
            p_hat[p_new_argsort[m]] = p_new[p_new_argsort[m]] * (1 - alpha * (m_star-1) / M) / ss
    
    return p_hat / p_hat.sum()

def train_Ada_OSMD(K, B_bar, data_list, eta_SGD, alpha, n_iter):
    M = len(data_list)
    d = data_list[0][0].shape[1]
    n = len(data_list[0][1])

    w = np.zeros(d)  # initialize w

    bar_a1 = bar_a1_est(B_bar, data_list)
    
    gamma = (alpha / M) * np.sqrt(8*K/(n_iter*bar_a1))

    lr_set = lr_set_fun(K, alpha, M, bar_a1, n_iter)  # get learning rates set
    E = len(lr_set)  # the length of learning rates set
    theta = (1+1/E) / (np.arange(1,E+1,1)*np.arange(2,E+2,1))  # initialize the weights of expert algorithm

    prob_sampling_experts = np.ones((E, M)) / M

    loss_list = []  # list of loss values
    regret_list = []  # list of regret

    for t in range(n_iter):
        # Compute meta algorithm sampling distribution
        prob_sampling = (np.expand_dims(theta, axis=1) * prob_sampling_experts).sum(0)
        prob_sampling /= prob_sampling.sum()  # normalize prob_sampling due to numerical error

        # Decide the batches for all clients
        batches = []
        for m in range(M):
            batch = list(np.random.choice(n, size=B_bar, replace=False))
            batches.append(batch)
        
        # choose client and make update
        G = np.zeros(d)  # sum of debiased gradients
        client_choose = np.random.choice(M, size=K, p=prob_sampling)  # choose client set with sampling distribution
        a_choose = np.zeros(K)  # feedback from environment
        for k in range(K):
            X, y = data_list[client_choose[k]]
            X_choose = X[batches[client_choose[k]],:]
            y_choose = y[batches[client_choose[k]]]
            g = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w))  # gradient of mini-batch
            a_choose[k] = ((1/M) * np.linalg.norm(g))**2
            g /= prob_sampling[client_choose[k]]  # debias the gradient
            G += g
        w -=  (eta_SGD / (M*K*B_bar)) * G  # make updates of parameter
        loss_list.append(loss_fun(w, data_list))

        # Compute regret
        g_mat = np.zeros((d,M))
        for m in range(M):
            X, y = data_list[m]
            X_choose = X[batches[m],:]
            y_choose = y[batches[m]]
            g_mat[:,m] = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w)) / B_bar
        g_norm_v = np.zeros(M)
        for m in range(M):
            g_norm_v[m] = np.linalg.norm(g_mat[:,m])
        regret_list.append( (((1/M)*g_norm_v) ** 2 / prob_sampling).sum() - ((1/M)*g_norm_v).sum()**2 )

        # compute gradients and losses for variance reduction
        l_hat = np.zeros(E)
        grad_l_hat = np.zeros((E, M))
        for e in range(E):
            for k in range(K):
                l_hat[e] += a_choose[k] / ( (K**2) * prob_sampling_experts[e, client_choose[k]] * prob_sampling[client_choose[k]] )
                grad_l_hat[e, client_choose[k]] = - a_choose[k] / ( (K**2) * (prob_sampling_experts[e, client_choose[k]]**2) * prob_sampling[client_choose[k]] )
        
        # make updates of sampling distribution
        for e in range(E):
            prob_sampling_experts[e,:] = Adaptive_OSMD_Expert(prob_sampling_experts[e,:], client_choose, grad_l_hat[e,:], lr_set[e], K, alpha)

        # make updates of the weights of expert algorithms
        if ( theta * np.exp(- gamma * l_hat ) ).sum() == 0:
            theta = np.ones(E) / E
        else:
            theta = theta * np.exp(- gamma * l_hat )  / ( theta * np.exp(- gamma * l_hat ) ).sum()
        theta /= theta.sum()
            
    return loss_list, regret_list

def train_Ada_OSMD_woreplace(K, B_bar, data_list, eta_SGD, alpha, n_iter):
    M = len(data_list)
    d = data_list[0][0].shape[1]
    n = len(data_list[0][1])

    w = np.zeros(d)  # initialize w

    bar_a1 = bar_a1_est(B_bar, data_list)
    
    gamma = (alpha / M) * np.sqrt(8*K/(n_iter*bar_a1))

    lr_set = lr_set_fun(K, alpha, M, bar_a1, n_iter)  # get learning rates set
    E = len(lr_set)  # the length of learning rates set
    theta = (1+1/E) / (np.arange(1,E+1,1)*np.arange(2,E+2,1))  # initialize the weights of expert algorithm

    prob_sampling_experts = np.ones((E, M)) / M

    loss_list = []  # list of loss values
    regret_list = []  # list of regret

    for t in range(n_iter):
        # Compute meta algorithm sampling distribution
        prob_sampling = (np.expand_dims(theta, axis=1) * prob_sampling_experts).sum(0)
        prob_sampling /= prob_sampling.sum()  # normalize prob_sampling due to numerical error

        # Decide the batches for all clients
        batches = []
        for m in range(M):
            batch = list(np.random.choice(n, size=B_bar, replace=False))
            batches.append(batch)
        
        # choose client and make update
        G = np.zeros(d)  # sum of debiased gradients
        prob_sampling /= prob_sampling.sum()  # normalize prob_sampling due to numerical error
        client_choose, prob_sampling_rescale = client_samp_replacement(M, K, prob_sampling)
        a_choose = np.zeros(K)  # feedback from environment
        g_chosen = np.zeros((K, d))
        for k in range(K):
            X, y = data_list[client_choose[k]]
            X_choose = X[batches[client_choose[k]],:]
            y_choose = y[batches[client_choose[k]]]
            g = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w))  # gradient of mini-batch
            a_choose[k] = ((1/M) * np.linalg.norm(g))**2
            g_chosen[k, :] = g
        for k in range(K):
            if k == 0:
                G += g_chosen[k, :] / prob_sampling_rescale[k, client_choose[k]]
            else:
                G += g_chosen[k, :] / prob_sampling_rescale[k, client_choose[k]] + g_chosen[0:k, :].sum(0)
        w -=  (eta_SGD / (M*K*B_bar)) * G  # make updates of parameter
        loss_list.append(loss_fun(w, data_list))

        # Compute regret
        g_mat = np.zeros((d,M))
        for m in range(M):
            X, y = data_list[m]
            X_choose = X[batches[m],:]
            y_choose = y[batches[m]]
            g_mat[:,m] = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w)) / B_bar
        g_norm_v = np.zeros(M)
        for m in range(M):
            g_norm_v[m] = np.linalg.norm(g_mat[:,m])
        regret_list.append( (((1/M)*g_norm_v) ** 2 / prob_sampling).sum() - ((1/M)*g_norm_v).sum()**2 )

        # compute gradients and losses for variance reduction
        l_hat = np.zeros(E)
        grad_l_hat = np.zeros((E, M))
        for e in range(E):
            for k in range(K):
                l_hat[e] += a_choose[k] / ( (K**2) * prob_sampling_experts[e, client_choose[k]] * prob_sampling[client_choose[k]] )
                grad_l_hat[e, client_choose[k]] = - a_choose[k] / ( (K**2) * (prob_sampling_experts[e, client_choose[k]]**2) * prob_sampling[client_choose[k]] )
        
        # make updates of sampling distribution
        for e in range(E):
            prob_sampling_experts[e,:] = Adaptive_OSMD_Expert(prob_sampling_experts[e,:], client_choose, grad_l_hat[e,:], lr_set[e], K, alpha)

        # make updates of the weights of expert algorithms
        if ( theta * np.exp(- gamma * l_hat ) ).sum() == 0:
            theta = np.ones(E) / E
        else:
            theta = theta * np.exp(- gamma * l_hat )  / ( theta * np.exp(- gamma * l_hat ) ).sum()
        theta /= theta.sum()
            
    return loss_list, regret_list

def bar_a1_est_renew(B_bar, data_list, w):
    M = len(data_list)
    d = data_list[0][0].shape[1]
    n = len(data_list[0][1])

    batches = []
    for m in range(M):
        batch = list(np.random.choice(n, size=B_bar, replace=False))
        batches.append(batch)

    g_mat = np.zeros((d,M))
    for m in range(M):
        X, y = data_list[m]
        X_choose = X[batches[m],:]
        y_choose = y[batches[m]]
        g_mat[:,m] = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w)) / B_bar

    g_norm_v = np.zeros(M)
    for m in range(M):
        g_norm_v[m] = np.linalg.norm(g_mat[:,m])

    return ((1/M)*g_norm_v.max())**2

def train_Ada_Doub_OSMD(K, B_bar, data_list, n_stop, eta_SGD, alpha):
    M = len(data_list)
    d = data_list[0][0].shape[1]
    n = len(data_list[0][1])

    b = 0  # initiliaze b
    w = np.zeros(d)  # initialize w

    loss_list = []  # list of loss values
    regret_list = []  # list of regret
    
    k_iter = 0  # number of iterations happened
    while True:
        b += 1
        n_iter = 2 ** (b-1)
        
        bar_a1 = bar_a1_est_renew(B_bar, data_list, w)
        lr_set = lr_set_fun(K, alpha, M, bar_a1, n_iter)  # get learning rates set
        E = len(lr_set)  # the length of learning rates set
        theta = (1+1/E) / (np.arange(1,E+1,1)*np.arange(2,E+2,1))  # initialize the weights of expert algorithms

        gamma = (alpha / M) * np.sqrt(8*K/(n_iter*bar_a1))
        
        # initialize the sampling distribution for expert algorithms
        if k_iter == 0:
            prob_sampling_experts = np.ones((E, M)) / M
        else:
            prob_sampling_experts = np.zeros((E, M))
            for e in range(E):
                prob_sampling_experts[e, :] = prob_sampling
    
        for t in range(n_iter):
            k_iter += 1
            
            # Compute meta algorithm sampling distribution
            prob_sampling = (np.expand_dims(theta, axis=1) * prob_sampling_experts).sum(0)
            prob_sampling /= prob_sampling.sum()  # normalize prob_sampling due to numerical error

            # Decide the batches for all clients
            batches = []
            for m in range(M):
                batch = list(np.random.choice(n, size=B_bar, replace=False))
                batches.append(batch)
        
            # choose client and make update
            G = np.zeros(d)  # sum of debiased gradients
            client_choose = np.random.choice(M, size=K, p=prob_sampling)  # choose client set with sampling distribution
            a_choose = np.zeros(K)  # feedback from environment
            for k in range(K):
                X, y = data_list[client_choose[k]]
                X_choose = X[batches[client_choose[k]],:]
                y_choose = y[batches[client_choose[k]]]
                g = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w))  # gradient of mini-batch
                a_choose[k] = ((1/M) * np.linalg.norm(g))**2
                g /= prob_sampling[client_choose[k]]  # debias the gradient
                G += g
            w -=  (eta_SGD / (M*K*B_bar)) * G  # make updates of parameter
            loss_list.append(loss_fun(w, data_list))

            # Compute regret
            g_mat = np.zeros((d,M))
            for m in range(M):
                X, y = data_list[m]
                X_choose = X[batches[m],:]
                y_choose = y[batches[m]]
                g_mat[:,m] = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w)) / B_bar
            g_norm_v = np.zeros(M)
            for m in range(M):
                g_norm_v[m] = np.linalg.norm(g_mat[:,m])
            regret_list.append( (((1/M)*g_norm_v) ** 2 / prob_sampling).sum() - ((1/M)*g_norm_v).sum()**2 )

            # compute gradients and losses for variance reduction
            l_hat = np.zeros(E)
            grad_l_hat = np.zeros((E, M))
            for e in range(E):
                for k in range(K):
                    l_hat[e] += a_choose[k] / ( (K**2) * prob_sampling_experts[e, client_choose[k]] * prob_sampling[client_choose[k]] )
                    grad_l_hat[e, client_choose[k]] = - a_choose[k] / ( (K**2) * (prob_sampling_experts[e, client_choose[k]]**2) * prob_sampling[client_choose[k]] )
        
            # make updates of sampling distribution
            for e in range(E):
                prob_sampling_experts[e,:] = Adaptive_OSMD_Expert(prob_sampling_experts[e,:], client_choose, grad_l_hat[e,:], lr_set[e], K, alpha)

            # make updates of the weights of expert algorithms
            if ( theta * np.exp(- gamma * l_hat ) ).sum() == 0:
                theta = np.ones(E) / E
            else:
                theta = theta * np.exp(- gamma * l_hat )  / ( theta * np.exp(- gamma * l_hat ) ).sum()
            theta /= theta.sum()
            
            # check if we should stop
            if k_iter >= n_stop:
                return loss_list, regret_list

def bar_a_est(B_bar, data_list):
    M = len(data_list)
    d = data_list[0][0].shape[1]
    n = len(data_list[0][1])

    w = np.zeros(d)  # initialize w
    batches = []
    for m in range(M):
        batch = list(np.random.choice(n, size=B_bar, replace=False))
        batches.append(batch)

    g_mat = np.zeros((d,M))
    for m in range(M):
        X, y = data_list[m]
        X_choose = X[batches[m],:]
        y_choose = y[batches[m]]
        g_mat[:,m] = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w)) / B_bar

    g_norm_v = np.zeros(M)
    for m in range(M):
        g_norm_v[m] = np.linalg.norm(g_mat[:,m])

    return (((1/M)*g_norm_v)**2).mean()

def train_MABS(K, B_bar, data_list, n_iter, eta_SGD, alpha):
    M = len(data_list)
    d = data_list[0][0].shape[1]
    n = len(data_list[0][1])
    
    loss_list = []  # list of loss values
    regret_list = []  # list of regret
    w = np.zeros(d)  # initialize w
    # initialize the sampling distribution as uniform distribution
    prob_sampling = np.ones(M) / M
    weights_log = np.zeros(M)  # initialize weights for MABS
    bar_a = bar_a_est(B_bar, data_list)
    delta = np.sqrt( (alpha**4)*np.log(M)/(n_iter*(M**5)*bar_a))
    for t in range(n_iter):
        # Decide the batches for all clients
        batches = []
        for m in range(M):
            batch = list(np.random.choice(n, size=B_bar, replace=False))
            batches.append(batch)
        
        # choose client and make update
        G = np.zeros(d)  # sum of debiased gradients
        prob_sampling /= prob_sampling.sum()  # normalize prob_sampling due to numerical error
        client_choose = np.random.choice(M, size=K, p=prob_sampling)  # choose client set with sampling distribution
        a_choose = np.zeros(K)  # feedback from environment
        for k in range(K):
            X, y = data_list[client_choose[k]]
            X_choose = X[batches[client_choose[k]],:]
            y_choose = y[batches[client_choose[k]]]
            g = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w))  # gradient of mini-batch
            a_choose[k] = ((1/M) * np.linalg.norm(g))**2
            g /= prob_sampling[client_choose[k]]  # debias the gradient
            G += g
        w -=  (eta_SGD / (M*K*B_bar)) * G  # make updates of parameter
        loss_list.append(loss_fun(w, data_list))

        # Compute regret
        g_mat = np.zeros((d,M))
        for m in range(M):
            X, y = data_list[m]
            X_choose = X[batches[m],:]
            y_choose = y[batches[m]]
            g_mat[:,m] = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w)) / B_bar
        g_norm_v = np.zeros(M)
        for m in range(M):
            g_norm_v[m] = np.linalg.norm(g_mat[:,m])
        regret_list.append( (((1/M)*g_norm_v) ** 2 / prob_sampling).sum() - ((1/M)*g_norm_v).sum()**2 )

        # make updates of sampling distribution
        for k in range(K):
            weights_log[client_choose[k]] = weights_log[client_choose[k]] + delta * a_choose[k] / (prob_sampling[client_choose[k]]**3)
        prob_sampling = (1 - alpha) * (np.exp(weights_log - weights_log.max()) / np.exp(weights_log - weights_log.max()).sum()) + alpha * np.ones(M) / M

    return loss_list, regret_list


def L_est(B_bar, data_list):
    M = len(data_list)
    d = data_list[0][0].shape[1]
    n = len(data_list[0][1])

    w = np.zeros(d)  # initialize w
    batches = []
    for m in range(M):
        batch = list(np.random.choice(n, size=B_bar, replace=False))
        batches.append(batch)

    L_v = np.zeros(M)
    for m in range(M):
        X, y = data_list[m]
        X_choose = X[batches[m],:]
        y_choose = y[batches[m]]
        L_v[m] = np.linalg.norm(y_choose - np.matmul(X_choose, w))**2 / (2*B_bar)
    
    return 1.5*(L_v.max()**2)

def train_VRB(K, B_bar, data_list, n_iter, eta_SGD, alpha):
    M = len(data_list)
    d = data_list[0][0].shape[1]
    n = len(data_list[0][1])
    
    loss_list = []  # list of loss values
    regret_list = []  # list of regret
    w = np.zeros(d)  # initialize w
    # initialize the sampling distribution as uniform distribution
    prob_sampling = np.ones(M) / M
    weights = np.zeros(M)  # initialize weights for MABS
    L = L_est(B_bar, data_list)
    for t in range(n_iter):
        # Decide the batches for all clients
        batches = []
        for m in range(M):
            batch = list(np.random.choice(n, size=B_bar, replace=False))
            batches.append(batch)
        
        # choose client and make update
        G = np.zeros(d)  # sum of debiased gradients
        prob_sampling /= prob_sampling.sum()  # normalize prob_sampling due to numerical error
        client_choose = np.random.choice(M, size=K, p=prob_sampling)  # choose client set with sampling distribution
        l_choose = np.zeros(K)  # feedback from environment
        for k in range(K):
            X, y = data_list[client_choose[k]]
            X_choose = X[batches[client_choose[k]],:]
            y_choose = y[batches[client_choose[k]]]
            g = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w))  # gradient of mini-batch
            l_choose[k] = np.linalg.norm(y_choose - np.matmul(X_choose, w))**2 / (2*B_bar)
            g /= prob_sampling[client_choose[k]]  # debias the gradient
            G += g
        w -=  (eta_SGD / (M*K*B_bar)) * G  # make updates of parameter
        loss_list.append(loss_fun(w, data_list))

        # Compute regret
        g_mat = np.zeros((d,M))
        for m in range(M):
            X, y = data_list[m]
            X_choose = X[batches[m],:]
            y_choose = y[batches[m]]
            g_mat[:,m] = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w)) / B_bar
        g_norm_v = np.zeros(M)
        for m in range(M):
            g_norm_v[m] = np.linalg.norm(g_mat[:,m])
        regret_list.append( (((1/M)*g_norm_v) ** 2 / prob_sampling).sum() - ((1/M)*g_norm_v).sum()**2 )

        # make updates of sampling distribution
        for k in range(K):
            weights[client_choose[k]] = weights[client_choose[k]] + l_choose[k]**2 / prob_sampling[client_choose[k]]
        prob_sampling = (1 - alpha) * np.sqrt(weights + L * M / alpha) / np.sqrt(weights + L * M / alpha).sum() + alpha * np.ones(M) / M

    return loss_list, regret_list

def client_samp_replacement(M, K, prob_sampling):
    # initialize client sampling distribution
    prob_sampling_rescale = np.zeros((K, M))
    prob_sampling_rescale[0, :] = prob_sampling

    client_chosen = []  # list of chosen clients
    
    for k in range(K):
        client_chosen.append( np.random.choice(M, size=1, p=prob_sampling_rescale[k, :])[0] )

        # Update client sampling probability
        if k < K-1:
            prob_sampling_rescale[k+1, :] = prob_sampling
            prob_sampling_rescale[k+1, client_chosen] = 0.0
            prob_sampling_rescale[k+1, :] = prob_sampling_rescale[k+1, :] / prob_sampling_rescale[k+1, :].sum()

    return client_chosen, prob_sampling_rescale

def train_Ada_Doub_OSMD_woreplace(K, B_bar, data_list, n_stop, eta_SGD, alpha, gamma):
    M = len(data_list)
    d = data_list[0][0].shape[1]
    n = len(data_list[0][1])

    b = 0  # initiliaze b
    w = np.zeros(d)  # initialize w
    
    loss_list = []  # list of loss values
    regret_list = []  # list of regret
    
    k_iter = 0  # number of iterations happened
    while True:
        b += 1
        n_iter = 2 ** (b-1)
        
        if b==1:
            bar_a1 = bar_a1_est(B_bar, data_list)
        else:
            bar_a1 = a_choose.max()
        
        lr_set = lr_set_fun(K, alpha, M, bar_a1, n_iter)  # get learning rates set
        E = len(lr_set)  # the length of learning rates set
        theta = (1+1/E) / (np.arange(1,E+1,1)*np.arange(2,E+2,1))  # initialize the weights of expert algorithms

        gamma = (alpha / M) * np.sqrt(8*K/(n_iter*bar_a1))
        
        # initialize the sampling distribution for expert algorithms
        if k_iter == 0:
            prob_sampling_experts = np.ones((E, M)) / M
        else:
            prob_sampling_experts = np.zeros((E, M))
            for e in range(E):
                prob_sampling_experts[e, :] = prob_sampling
    
        for t in range(n_iter):
            k_iter += 1
            
            # Compute meta algorithm sampling distribution
            prob_sampling = (np.expand_dims(theta, axis=1) * prob_sampling_experts).sum(0)

            # Decide the batches for all clients
            batches = []
            for m in range(M):
                batch = list(np.random.choice(n, size=B_bar, replace=False))
                batches.append(batch)
        
            # choose client and make update
            G = np.zeros(d)  # sum of debiased gradients
            prob_sampling /= prob_sampling.sum()  # normalize prob_sampling due to numerical error
            client_choose, prob_sampling_rescale = client_samp_replacement(M, K, prob_sampling)
            a_choose = np.zeros(K)  # feedback from environment
            g_chosen = np.zeros((K, d))
            for k in range(K):
                X, y = data_list[client_choose[k]]
                X_choose = X[batches[client_choose[k]],:]
                y_choose = y[batches[client_choose[k]]]
                g = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w))  # gradient of mini-batch
                a_choose[k] = ((1/M) * np.linalg.norm(g))**2
                g_chosen[k, :] = g
            for k in range(K):
                if k == 0:
                    G += g_chosen[k, :] / prob_sampling_rescale[k, client_choose[k]]
                else:
                    G += g_chosen[k, :] / prob_sampling_rescale[k, client_choose[k]] + g_chosen[0:k, :].sum(0)
            w -=  (eta_SGD / (M*K*B_bar)) * G  # make updates of parameter
            loss_list.append(loss_fun(w, data_list))

            # Compute regret
            g_mat = np.zeros((d,M))
            for m in range(M):
                X, y = data_list[m]
                X_choose = X[batches[m],:]
                y_choose = y[batches[m]]
                g_mat[:,m] = -np.matmul(X_choose.T, y_choose - np.matmul(X_choose, w)) / B_bar
            g_norm_v = np.zeros(M)
            for m in range(M):
                g_norm_v[m] = np.linalg.norm(g_mat[:,m])
            regret_list.append( (((1/M)*g_norm_v) ** 2 / prob_sampling).sum() - ((1/M)*g_norm_v).sum()**2 )

            # compute gradients and losses
            l_hat = np.zeros(E)
            grad_l_hat = np.zeros((E, M))
            for e in range(E):
                for k in range(K):
                    l_hat[e] += a_choose[k] / ( (K**2) * prob_sampling_experts[e, client_choose[k]] * prob_sampling[client_choose[k]] )
                    grad_l_hat[e, client_choose[k]] = - a_choose[k] / ( (K**2) * (prob_sampling_experts[e, client_choose[k]]**2) * prob_sampling[client_choose[k]] )
        
            # make updates of sampling distribution
            for e in range(E):
                prob_sampling_experts[e,:] = Adaptive_OSMD_Expert(prob_sampling_experts[e,:], client_choose, grad_l_hat[e,:], lr_set[e], K, alpha)

            # make updates of the weights of expert algorithms
            if ( theta * np.exp(- gamma * l_hat ) ).sum() == 0:
                theta = np.ones(E) / E
            else:
                theta = theta * np.exp(- gamma * l_hat )  / ( theta * np.exp(- gamma * l_hat ) ).sum()
            theta /= theta.sum()
            
            # check if we should stop
            if k_iter >= n_stop:
                return loss_list, regret_list
