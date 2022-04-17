import numpy as np
import random
import torch
import torchvision
import torch.nn as nn
import time
from torch.utils import data
from d2l import torch as d2l
from matplotlib import pyplot as plt
from IPython import display


#################### Data Pipeline ####################

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
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=dataloader_worker_count))


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


#################### Test & Train ####################

def evaluate_accuracy(net, data_iter, device=torch.device('cpu')):
    '''
    evaluate accuracy of a model with given dataset
    '''
    net.eval() # switch to evaluation mode for Dropout, BatchNorm etc. layers
    if not device:  # 查询第一个参数所在的第一个设备
        device = list(net.collect_params().values())[0].list_ctx()[0]
    acc_sum, n = torch.tensor([0], dtype=torch.float32, device=device), 0
    for X, y in data_iter:
        # Copy to device
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y = y.long()
            acc_sum += torch.sum((torch.argmax(net(X), dim=1) == y))
            n += y.shape[0]
    return acc_sum.item() / n


def train_cnn_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    # init params
    net.apply(init_weights)
    # copy to device
    print('training on', device)
    net.to(device)
    # define optimizer & loss function
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    timer = Timer()
    num_batches = len(train_iter)
    for epoch in range(num_epochs):
        # train loss, train acc sum, sample count
        metric = d2l.Accumulator(3)
        # switch to train mode
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                y = y.long()
                acc_sum = torch.sum((torch.argmax(net(X), dim=1) == y))
                n = y.shape[0]
                metric.add(l * X.shape[0], acc_sum, n)
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy(net, test_iter, device)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


#################### Plotting ####################

# from d2l
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """A utility function to set matplotlib axes"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend: axes.legend(legend)
    axes.grid()


# from d2l
class Animator(object):
    def __init__(self, xlabel=None, ylabel=None, legend=[], xlim=None,
                 ylim=None, xscale='linear', yscale='linear', fmts=None,
                 nrows=1, ncols=1, figsize=(3.5, 2.5)):
        """Incrementally plot multiple lines."""
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1: self.axes = [self.axes,]
        # use a lambda to capture arguments
        self.config_axes = lambda : set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """Add multiple data points into the figure."""
        if not hasattr(y, "__len__"): y = [y]
        n = len(y)
        if not hasattr(x, "__len__"): x = [x] * n
        if not self.X: self.X = [[] for _ in range(n)]
        if not self.Y: self.Y = [[] for _ in range(n)]
        if not self.fmts: self.fmts = ['-'] * n
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

#################### Accumulator ####################

# from d2l
class Accumulator(object):
    """
    Sum a list of numbers over time
    from: https://github.com/dsgiitr/d2l-pytorch/blob/master/d2l/base.py
    """
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + b for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0] * len(self.data)
    def __getitem__(self, i):
        return self.data[i]


#################### Timer ####################

# from d2l
class Timer(object):
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()
        
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
    
    def stop(self):
        """Stop the timer and record the time in a list"""
        self.times.append(time.time() - self.start_time)
        return self.times[-1]
        
    def avg(self):
        """Return the average time"""
        return sum(self.times)/len(self.times)
    
    def sum(self):
        """Return the sum of time"""
        return sum(self.times)
        
    def cumsum(self):
        """Return the accumuated times"""
        return np.array(self.times).cumsum().tolist()