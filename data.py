import torch
from torch.utils.data import DataLoader


def generate_random_data(d, N):
    X = torch.rand(N, d)
    Y = torch.rand(N, 1)
    Y = torch.round(Y)
    return X, Y


def generate_perfect_data(d, N):
    Y = torch.rand(N, 1)
    Y = torch.round(Y)
    X = torch.zeros(N, d) + Y
    return X, Y


def split_data_into_datasets(X, Y, split=0.8):
    train_size = int(split * X.shape[0])
    train_dataset = torch.utils.data.TensorDataset(X[:train_size], Y[:train_size])
    test_dataset = torch.utils.data.TensorDataset(X[train_size:], Y[train_size:])
    return train_dataset, test_dataset


def load_tensor_to_dataloader(X, Y, batch_size, split=0.8):
    train_data, test_data = split_data_into_datasets(X, Y, split)
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader