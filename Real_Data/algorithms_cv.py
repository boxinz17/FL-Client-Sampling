import numpy as np
import torch
import torch.nn as nn
import time

def model_eval(models_list, train_loader_list, val_loader_list, device):
    # Model evaluation function: Given parameters w, compute
    # the mean loss and validation accuracy of all clients
    # params:
    #   w: the list of length 2, w[0] is the weights and
    #   w[1] is the bias term
    #   train_loader_list: list of training data loader
    #   val_loader_list: list of validation data loader
    #   device: the device for storage and computation
    M = len(models_list)

    # Define cross entropy loss functions
    loss_CE_fn = nn.CrossEntropyLoss()
    # Define softmanx function
    softmax_model = nn.Softmax(dim=1)

    loss = 0.0  # training loss
    right_count = 0  # number of correct predictions
    n_train = 0  # number of training samples
    n_val = 0  # number of validation samples
    for m in range(M):
        # set up model
        model = models_list[m]
        
        # set up training dataset
        train_loader = train_loader_list[m]

        # Compute the local loss
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            n_train += len(labels)
             
            # Compute loss of model
            with torch.no_grad():
                outs_local = model(imgs.view(imgs.shape[0], -1))
                loss_local = loss_CE_fn(outs_local, labels)
            
                # Update training loss
                loss += loss_local.item()

        # Set up validation dataset
        val_loader = val_loader_list[m]
        
        # Begin computing the accuracy
        for imgs, labels in val_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            n_val += len(labels)
            
            # Compute the number of right cases
            with torch.no_grad():
                outs = softmax_model(model(imgs.view(imgs.shape[0], -1)))
                outs = torch.argmax(outs, dim=1)
                
                # Update correct prediction numbers
                right_count += (outs == labels).sum().item()
    
    # Compute average training loss and validation accuracy
    loss /= n_train
    accu = right_count / n_val
    
    return loss, accu

def train_uniform(train_loader_list, train_list, val_loader_list, K, B_bar, n_commun, lr, device):
    M = len(train_list)

    loss_CE_fn = nn.CrossEntropyLoss()  # define cross-entropy loss

    # initialize w
    w_list = [torch.zeros(10, 784).to(device), torch.zeros(10).to(device)]
    
    # initialize list of models
    models_list = []
    for m in range(M):
        model = nn.Linear(784, 10).to(device=device)
        with torch.no_grad():
            model.weight[:] = torch.zeros(10, 784).to(device)
            model.bias[:] = torch.zeros(10).to(device)
        models_list.append(model)
    
    # Compute the total number of training samples
    n_train_total = 0
    for m in range(M):
        n_train_total += len(train_list[m])
    
    # Compute the weights for each client in overall loss function
    lambda_train = np.zeros(M)
    for m in range(M):
        lambda_train[m] = len(train_list[m]) / n_train_total
    lambda_train = torch.from_numpy(lambda_train).to(device)  # move it to chosen device
    
    loss_list = []  # list of training losses
    accu_list = []  # list of validation accuracies
    
    start_time = time.time()
    for t in range(n_commun):
        # set the sampling distribution for choosing clients
        prob_sampling =  np.ones(M) / M
        prob_sampling /= prob_sampling.sum()

        # choose client and make update
        client_choose = np.random.choice(M, size=K, p=prob_sampling)  # choose client set with sampling distribution
        # the average of gradients of chosen clients
        avg_grad = [torch.zeros(10, 784).to(device), torch.zeros(10).to(device)]
        
        for k in range(K):
            n_train = len(train_list[client_choose[k]])  # the number of training samples of the client
            B_bar = min(n_train, B_bar)  # decide the mini-batch size
            
            local_model = models_list[client_choose[k]]  # load local model
            train_local = train_list[client_choose[k]]  # load local training data
            
            # implement FedSGD
            batch = list(np.random.choice(n_train, size=B_bar, replace=False))  # decide the samples in bacth for mini-batch sgd
            loss_local = 0.0
            for j in range(B_bar):
                img, label = train_local[batch[j]]
                img = img.to(device=device)
                label = torch.Tensor([label]).to(device=device).long()

                # Compute loss of each sample
                out_local = local_model(img.view(1,-1))
                loss_local += loss_CE_fn(out_local, label)
            # Average the loss by mini-batch size
            loss_local /= B_bar

            # Compute gradients
            loss_local.backward()

            # Compute the weighted gradient and add it for average
            for i, param in enumerate(local_model.parameters()):
                avg_grad[i] += param.grad * lambda_train[client_choose[k]] * M

            # zero out grad
            local_model.zero_grad()
        
        # Divide all gradients by K
        for i in range(2):
            avg_grad[i] /= K

        # make update of w
        for i in range(2):
            w_list[i] -= lr * avg_grad[i]

        # make update of the models
        for m in range(M):
            with torch.no_grad():
                models_list[m].weight[:] = w_list[0].clone()
                models_list[m].bias[:] = w_list[1].clone()

        # Evalue the current model
        loss_train, accu = model_eval(models_list, train_loader_list, val_loader_list, device)
        loss_list.append(loss_train)
        accu_list.append(accu)

        if (t+1) % 100 == 0 or t+1 == 1:
            end_time = time.time()
            print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | test accuracy: {}".format(t+1, end_time-start_time, loss_list[-1], accu_list[-1]))

    return loss_list, accu_list

def optimal_dist(B_bar, train_list, models_list, device):
    M = len(train_list)

    loss_CE_fn = nn.CrossEntropyLoss()  # define cross-entropy loss

    # Compute the total number of training samples
    n_train_total = 0
    for m in range(M):
        n_train_total += len(train_list[m])
    
    # Compute the weights for each client in overall loss function
    lambda_train = np.zeros(M)
    for m in range(M):
        lambda_train[m] = len(train_list[m]) / n_train_total
    lambda_train = torch.from_numpy(lambda_train).to(device)  # move it to chosen device

    # choose client and make update
    g_norm_v = np.zeros(M)
    for m in range(M):
        local_model = models_list[m]  # load local model
        train_local = train_list[m]  # load local training data
            
        # implement FedSGD
        n_train = len(train_list[m])  # the number of training samples of the client
        B_bar = min(n_train, B_bar)  # decide the mini-batch size

        batch = list(np.random.choice(n_train, size=B_bar, replace=False))  # decide the samples in bacth for mini-batch sgd
        loss_local = 0.0
        for j in range(B_bar):
            img, label = train_local[batch[j]]
            img = img.to(device=device)
            label = torch.Tensor([label]).to(device=device).long()

            # Compute loss of each sample
            out_local = local_model(img.view(1,-1))
            loss_local += loss_CE_fn(out_local, label)
        loss_local /= B_bar

        # Compute gradients
        loss_local.backward()

        # Compute the norm of gradient
        grad_norm = np.zeros(2)
        for i, param in enumerate(local_model.parameters()):
            grad_norm[i] = torch.norm(param.grad).to('cpu')
        g_norm_v[m] = np.sqrt((grad_norm**2).sum())

        # zero out grad
        local_model.zero_grad()
    
    asqrt_v = lambda_train * grad_norm
    prob = asqrt_v / asqrt_v.sum()
    for m in range(M):
        if prob[m] <= 1e-10:
            prob[m] = 0.0
    prob = prob / prob.sum()

    return prob

def train_optimal(train_loader_list, train_list, val_loader_list, K, B_bar, n_commun, lr, device):
    M = len(train_list)

    loss_CE_fn = nn.CrossEntropyLoss()  # define cross-entropy loss

    # initialize w
    w_list = [torch.zeros(10, 784).to(device), torch.zeros(10).to(device)]
    
    # initialize list of models
    models_list = []
    for m in range(M):
        model = nn.Linear(784, 10).to(device=device)
        with torch.no_grad():
            model.weight[:] = torch.zeros(10, 784).to(device)
            model.bias[:] = torch.zeros(10).to(device)
        models_list.append(model)

    # Compute the total number of training samples
    n_train_total = 0
    for m in range(M):
        n_train_total += len(train_list[m])
    
    # Compute the weights for each client in overall loss function
    lambda_train = np.zeros(M)
    for m in range(M):
        lambda_train[m] = len(train_list[m]) / n_train_total
    lambda_train = torch.from_numpy(lambda_train).to(device)  # move it to chosen device
    
    loss_list = []  # list of training losses
    accu_list = []  # list of validation accuracies
    start_time = time.time()
    for t in range(n_commun):
        # set the sampling distribution for choosing clients
        prob_sampling =  optimal_dist(B_bar, train_list, models_list, device)
        prob_sampling /= prob_sampling.sum()

        # choose client and make update
        client_choose = np.random.choice(M, size=K, p=prob_sampling)  # choose client set with sampling distribution
        # the average of gradients of chosen clients
        avg_grad = [torch.zeros(10, 784).to(device), torch.zeros(10).to(device)]
        for k in range(K):
            n_train = len(train_list[client_choose[k]])
            B_bar = min(n_train, B_bar)

            local_model = models_list[client_choose[k]]  # load local model
            train_local = train_list[client_choose[k]]  # load local training data
            
            # implement FedSGD
            batch = list(np.random.choice(n_train, size=B_bar, replace=False))  # decide the samples in bacth for mini-batch sgd
            loss_local = 0.0
            for j in range(B_bar):
                img, label = train_local[batch[j]]
                img = img.to(device=device)
                label = torch.Tensor([label]).to(device=device).long()

                # Compute loss of each sample
                out_local = local_model(img.view(1,-1))
                loss_local += loss_CE_fn(out_local, label)
            # Average the loss by mini-batch size
            loss_local /= B_bar

            # Compute gradients
            loss_local.backward()

            # Compute the weighted gradient and add it for average
            for i, param in enumerate(local_model.parameters()):
                avg_grad[i] += param.grad * lambda_train[client_choose[k]] / prob_sampling[client_choose[k]]

            # zero out grad
            local_model.zero_grad()
        
        # Divide all gradients by K
        for i in range(2):
            avg_grad[i] /= K

        # make update of w
        for i in range(2):
            w_list[i] -= lr * avg_grad[i]

        # make update of the models
        for m in range(M):
            with torch.no_grad():
                models_list[m].weight[:] = w_list[0].clone()
                models_list[m].bias[:] = w_list[1].clone()

        # Evalue the current model
        loss_train, accu = model_eval(models_list, train_loader_list, val_loader_list, device)
        loss_list.append(loss_train)
        accu_list.append(accu)

        if (t+1) % 100 == 0 or t+1 == 1:
            end_time = time.time()
            print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | test accuracy: {}".format(t+1, end_time-start_time, loss_list[-1], accu_list[-1]))

    return loss_list, accu_list

def bar_a1_est(B_bar, train_list, device):
    M = len(train_list)

    # initialize list of models
    models_list = []
    for m in range(M):
        model = nn.Linear(784, 10).to(device=device)
        with torch.no_grad():
            model.weight[:] = torch.zeros(10, 784).to(device)
            model.bias[:] = torch.zeros(10).to(device)
        models_list.append(model)

    loss_CE_fn = nn.CrossEntropyLoss()  # define cross-entropy loss

    g_norm_v = np.zeros(M)
    for m in range(M):
        local_model = models_list[m]  # load local model
        train_local = train_list[m]  # load local training data
        
        # implement FedSGD
        n_train = len(train_list[m])
        B_bar = min(n_train, B_bar)
        
        batch = list(np.random.choice(n_train, size=B_bar, replace=False))  # decide the samples in bacth for mini-batch sgd
        loss_local = 0.0
        for j in range(B_bar):
            img, label = train_local[batch[j]]
            img = img.to(device=device)
            label = torch.Tensor([label]).to(device=device).long()

            # Compute loss of each sample
            out_local = local_model(img.view(1,-1))
            loss_local += loss_CE_fn(out_local, label)
        loss_local /= B_bar

        # Compute gradients
        loss_local.backward()

        # Compute the norm of gradient and add it for average
        grad_norm = np.zeros(2)
        for i, param in enumerate(local_model.parameters()):
            grad_norm[i] = torch.norm(param.grad).to('cpu')
        g_norm_v[m] = np.sqrt((grad_norm**2).sum())

        # zero out grad
        local_model.zero_grad()

    return ((1/M)*g_norm_v.max())**2

def bar_a1_est_renew(B_bar, train_list, models_list, device):
    M = len(train_list)

    loss_CE_fn = nn.CrossEntropyLoss()  # define cross-entropy loss

    g_norm_v = np.zeros(M)
    for m in range(M):
        local_model = models_list[m]  # load local model
        train_local = train_list[m]  # load local training data
            
        # implement FedSGD
        n_train = len(train_list[m])
        B_bar = min(n_train, B_bar)

        batch = list(np.random.choice(n_train, size=B_bar, replace=False))  # decide the samples in bacth for mini-batch sgd
        loss_local = 0.0
        for j in range(B_bar):
            img, label = train_local[batch[j]]
            img = img.to(device=device)
            label = torch.Tensor([label]).to(device=device).long()

            # Compute loss of each sample
            out_local = local_model(img.view(1,-1))
            loss_local += loss_CE_fn(out_local, label)
        loss_local /= B_bar

        # Compute gradients
        loss_local.backward()

        # Compute the norm of gradient and add it for average
        grad_norm = np.zeros(4)
        for i, param in enumerate(local_model.parameters()):
            grad_norm[i] = torch.norm(param.grad).to('cpu')
        g_norm_v[m] = np.sqrt((grad_norm**2).sum())

        # zero out grad
        local_model.zero_grad()

    return ((1/M)*g_norm_v.max())**2

def lr_set_fun(K, alpha, M, bar_a1, n_commun):
    E = np.ceil( 0.5 * np.log2( (np.log(M)+4*(n_commun-1)*np.log(M/alpha)) / np.log(M) ) ) + 1
    E = int(E)
    lr_set = []
    for e in range(E):
        lr_set.append( 2**e * (K**2 * alpha**3 / (M**3 * bar_a1)) * np.sqrt(np.log(M)/(2*n_commun)) )
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

def train_Ada_OSMD(train_loader_list, train_list, val_loader_list, alpha, K, B_bar, n_commun, lr, device):
    M = len(train_list)

    loss_CE_fn = nn.CrossEntropyLoss()  # define cross-entropy loss

    # initialize w
    w_list = [torch.zeros(10, 784).to(device), torch.zeros(10).to(device)]
    
    # initialize list of models
    models_list = []
    for m in range(M):
        model = nn.Linear(784, 10).to(device=device)
        with torch.no_grad():
            model.weight[:] = torch.zeros(10, 784).to(device)
            model.bias[:] = torch.zeros(10).to(device)
        models_list.append(model)

    # Compute the total number of training samples
    n_train_total = 0
    for m in range(M):
        n_train_total += len(train_list[m])
    
    # Compute the weights for each client in overall loss function
    lambda_train = np.zeros(M)
    for m in range(M):
        lambda_train[m] = len(train_list[m]) / n_train_total
    lambda_train = torch.from_numpy(lambda_train).to(device)  # move it to chosen device
    
    bar_a1 = bar_a1_est_renew(B_bar, train_list, models_list, device)
    gamma = (alpha / M) * np.sqrt(8*K/(n_commun*bar_a1))

    lr_set = lr_set_fun(K, alpha, M, bar_a1, n_commun)  # get learning rates set
    E = len(lr_set)  # the length of learning rates set
    theta = (1+1/E) / (np.arange(1,E+1,1)*np.arange(2,E+2,1))  # initialize the weights of expert algorithms

    prob_sampling_experts = np.ones((E, M)) / M

    loss_list = []  # list of training losses
    accu_list = []  # list of validation accuracies
    start_time = time.time()

    for t in range(n_commun):
        # Compute meta algorithm sampling distribution
        prob_sampling = (np.expand_dims(theta, axis=1) * prob_sampling_experts).sum(0)
        prob_sampling /= prob_sampling.sum()  # normalize prob_sampling due to numerical error

        # choose client and make update
        client_choose = np.random.choice(M, size=K, p=prob_sampling)  # choose client set with sampling distribution
        a_choose = np.zeros(K)  # feedback from environment
        # the average of gradients of chosen clients
        avg_grad = [torch.zeros(10, 784).to(device), torch.zeros(10).to(device)]
        
        for k in range(K):
            local_model = models_list[client_choose[k]]  # load local model
            train_local = train_list[client_choose[k]]  # load local training data
            
            # implement FedSGD
            n_train = len(train_list[client_choose[k]])  # the number of training samples of the client
            B_bar = min(n_train, B_bar)  # decide the mini-batch size

            batch = list(np.random.choice(n_train, size=B_bar, replace=False))  # decide the samples in bacth for mini-batch sgd
            loss_local = 0.0
            for j in range(B_bar):
                img, label = train_local[batch[j]]
                img = img.to(device=device)
                label = torch.Tensor([label]).to(device=device).long()

                # Compute loss of each sample
                out_local = local_model(img.view(1,-1))
                loss_local += loss_CE_fn(out_local, label)
           # Average the loss by mini-batch size
            loss_local /= B_bar

            # Compute gradients
            loss_local.backward()

            # Compute the norm of gradient and add it for average
            grad_norm = np.zeros(2)
            for i, param in enumerate(local_model.parameters()):
                avg_grad[i] += param.grad * lambda_train[client_choose[k]] / prob_sampling[client_choose[k]]
                grad_norm[i] = torch.norm(param.grad).to('cpu')
            a_choose[k] = (lambda_train[client_choose[k]]**2) * ((grad_norm**2).sum())

            # zero out grad
            local_model.zero_grad()
            
        # Divide all gradients by K
        for i in range(2):
            avg_grad[i] /= K

        # make update of w
        for i in range(2):
            w_list[i] -= lr * avg_grad[i]

        # make update of the models
        for m in range(M):
            with torch.no_grad():
                models_list[m].weight[:] = w_list[0].clone()
                models_list[m].bias[:] = w_list[1].clone()

        # Evalue the current model
        loss_train, accu = model_eval(models_list, train_loader_list, val_loader_list, device)
        loss_list.append(loss_train)
        accu_list.append(accu)

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

        if (t+1) % 100 == 0 or t == 0:
            end_time = time.time()
            print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | test accuracy: {}".format(t+1, end_time-start_time, loss_list[-1], accu_list[-1]))

    return loss_list, accu_list

def train_AdaDoub_OSMD(train_loader_list, train_list, val_loader_list, alpha, K, B_bar, n_commun, lr, device):
    M = len(train_list)

    loss_CE_fn = nn.CrossEntropyLoss()  # define cross-entropy loss

    # initialize w
    w_list = [torch.zeros(10, 784).to(device), torch.zeros(10).to(device)]
    
    # initialize list of models
    models_list = []
    for m in range(M):
        model = nn.Linear(784, 10).to(device=device)
        with torch.no_grad():
            model.weight[:] = torch.zeros(10, 784).to(device)
            model.bias[:] = torch.zeros(10).to(device)
        models_list.append(model)

    # Compute the total number of training samples
    n_train_total = 0
    for m in range(M):
        n_train_total += len(train_list[m])
    
    # Compute the weights for each client in overall loss function
    lambda_train = np.zeros(M)
    for m in range(M):
        lambda_train[m] = len(train_list[m]) / n_train_total
    lambda_train = torch.from_numpy(lambda_train).to(device)  # move it to chosen device

    loss_list = []  # list of training losses
    accu_list = []  # list of validation accuracies
    start_time = time.time()

    k_iter = 0  # number of iterations happened
    b = 0
    while True:
        b += 1
        n_iter = 2 ** (b-1)

        bar_a1 = bar_a1_est_renew(B_bar, train_list, models_list, device)
        gamma = (alpha / M) * np.sqrt(8*K/(n_iter*bar_a1))

        lr_set = lr_set_fun(K, alpha, M, bar_a1, n_iter)  # get learning rates set
        E = len(lr_set)  # the length of learning rates set
        theta = (1+1/E) / (np.arange(1,E+1,1)*np.arange(2,E+2,1))  # initialize the weights of expert algorithms

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

            # choose client and make update
            client_choose = np.random.choice(M, size=K, p=prob_sampling)  # choose client set with sampling distribution
            a_choose = np.zeros(K)  # feedback from environment
            # the average of gradients of chosen clients
            avg_grad = [torch.zeros(10, 784).to(device), torch.zeros(10).to(device)]
            for k in range(K):
                local_model = models_list[client_choose[k]]  # load local model
                train_local = train_list[client_choose[k]]  # load local training data
            
                # implement FedSGD
                n_train = len(train_list[client_choose[k]])
                B_bar = min(n_train, B_bar)

                batch = list(np.random.choice(n_train, size=B_bar, replace=False))  # decide the samples in bacth for mini-batch sgd
                loss_local = 0.0
                for j in range(B_bar):
                    img, label = train_local[batch[j]]
                    img = img.to(device=device)
                    label = torch.Tensor([label]).to(device=device).long()

                    # Compute loss of each sample
                    out_local = local_model(img.view(1,-1))
                    loss_local += loss_CE_fn(out_local, label)
                # Average the loss by mini-batch size
                loss_local /= B_bar

                # Compute gradients
                loss_local.backward()

                # Compute the norm of gradient and add it for average
                grad_norm = np.zeros(2)
                for i, param in enumerate(local_model.parameters()):
                    avg_grad[i] += param.grad * lambda_train[client_choose[k]] / prob_sampling[client_choose[k]]
                    grad_norm[i] = torch.norm(param.grad).to('cpu')
                a_choose[k] = (lambda_train[client_choose[k]]**2) * ((grad_norm**2).sum())

                # zero out grad
                local_model.zero_grad()
            
            # Divide all gradients by K
            for i in range(2):
                avg_grad[i] /= K

            # make update of w
            for i in range(2):
                w_list[i] -= lr * avg_grad[i]

            # make update of the models
            for m in range(M):
                with torch.no_grad():
                    models_list[m].weight[:] = w_list[0].clone()
                    models_list[m].bias[:] = w_list[1].clone()

            # Evalue the current model
            loss_train, accu = model_eval(models_list, train_loader_list, val_loader_list, device)
            loss_list.append(loss_train)
            accu_list.append(accu)

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

            if k_iter % 100 == 0 or k_iter == 1:
                end_time = time.time()
                print("commun times: {} | time pass : {:.2f}(s) | training loss is: {} | test accuracy: {}".format(k_iter, end_time-start_time, loss_list[-1], accu_list[-1]))

            # check if we should stop
            if k_iter >= n_commun:
                return loss_list, accu_list