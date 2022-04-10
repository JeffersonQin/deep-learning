import random
import torch
import torchvision
from torch.utils import data
from d2l import torch as d2l
from matplotlib import pyplot as plt


# 进行分装，增加 Resize 功能
def load_data_fashion_mnist(batch_size, dataloader_worker_count, resize=None):
    trans = [torchvision.transforms.ToTensor()]
    if resize:
        trans.insert(0, torchvision.transforms.Resize(resize))
    trans = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=dataloader_worker_count),
            data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=dataloader_worker_count))


def load_data_iter(features, labels, batch_size):
    '''
    return a data iterator with mini-batch feature
    '''
    num = len(features)
    indices = list(range(num))
    # randomize
    random.shuffle(indices)
    for i in range(0, num, batch_size):
        # use min to prevent crossing border
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num)])
        yield features[batch_indices], labels[batch_indices]
