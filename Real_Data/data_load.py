import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets

def data_prepare(dataset_name, n_devices, n_train_list, n_val, batch_size=64, rd_seed=111):
    """
    Return the train_loader_list, devices_train_list, val_loader_list and devices_val_list
    based on the dataset_name.
    """
    # Input: 
    #   dataset_name: A string, should be one of
    #   {"MNIST", "KMNIST", "FMNIST"}
    #   n_devices: number of devices, integer
    #   n_train_list: number of samples per device with corresponding proportion
    #   n_val: number of samples per device for validation, integer
    #   batch_size: batch size for creating mini batched
    #               train loader
    #   rd_seed: random seed
    # Return:
    #   (train_loader_list, devices_train_list, val_loader_list, devices_val_list)
    np.random.seed(rd_seed)

    if dataset_name == "MNIST":
        data_path = 'data/mnist/'
        transform_data = datasets.MNIST(
            data_path, train=True, download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
            ]))
    elif dataset_name == "KMNIST":
        data_path = 'data/kmnist/'
        transform_data = datasets.KMNIST(
            data_path, train=True, download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
            ]))
    elif dataset_name == "FMNIST":
        data_path = 'data/fmnist/'
        transform_data = datasets.FashionMNIST(
            data_path, train=True, download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
            ]))
        
    devices_train_list = []  # list of training data for devices
    train_loader_list = []  # list of train loader for devices
    devices_val_list = []  # list of validation data for devices
    val_loader_list = []  # list of validation loader for devices

    n_data = len(transform_data)
    for m in range(n_devices):
        # Decide the number of training samples
        n_train = np.random.choice(a=n_train_list[0], size=1, p=n_train_list[1])[0]

        # Choose training and validation data
        samples_choose = np.random.choice(n_data, size=(n_train + n_val), replace=False)
        device_data = []
        for k in range(len(samples_choose)):
            device_data.append(transform_data[samples_choose[k]])
        devices_train_list.append(device_data[:n_train])
        train_loader = torch.utils.data.DataLoader(device_data[:n_train], batch_size=batch_size, shuffle=False)
        train_loader_list.append(train_loader)
        devices_val_list.append(device_data[n_train:])
        val_loader = torch.utils.data.DataLoader(device_data[n_train:], batch_size=batch_size, shuffle=False)
        val_loader_list.append(val_loader)
            
    return train_loader_list, devices_train_list, val_loader_list, devices_val_list