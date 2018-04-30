import os
import timeit

import pymp
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch import multiprocessing as mp
import redis
import sgd
from redis import StrictRedis
from rediscluster import StrictRedisCluster

from common_functions import push_params_redis, get_shapes, get_params_redis, set_params, push_params_redis_init, \
    check_param_exists


def train(args, model):
    # startup_nodes = [{"host": "127.0.0.1", "port": "30001"},{"host": "127.0.0.1", "port": "30002"}]
    db = redis.ConnectionPool(host='localhost', port=6379, db=0)
    db = redis.StrictRedis(connection_pool=db)
    # db = StrictRedisCluster(startup_nodes=startup_nodes, decode_responses=True)

    params_exists = check_param_exists(model, db)
    if not params_exists:
        push_params_redis_init(model,db)
    shapes = get_shapes(model)

    # Print total number of processes
    print(args.num_processes)

    # Using pymp to parallelise the training
    epoch_time = 0
    with pymp.Parallel(args.num_processes) as p:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, num_workers=1)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.batch_size, shuffle=True, num_workers=1)
        # Optimizer declaration
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        for epoch in range(args.epochs):
            epoch_start_time = timeit.default_timer()
            loss = train_process(p.thread_num, optimizer, train_loader, model, args, shapes, db, epoch)
            epoch_time = epoch_time + timeit.default_timer() - epoch_start_time
            # if p.thread_num :
            print('PID{}\tTrain Epoch: {}\t time: {} \tLoss: {:.6f}'.format(p.thread_num, epoch, epoch_time, loss))
            # test_epoch(model, test_loader)


def shuffle_tensor(tensor):
    return torch.randperm(tensor.size(0)).long()


def train_process(thread_num, optimizer, train_loader, model, args, shapes, db, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):

        set_params(optimizer, get_params_redis(db, shapes))
        model.train()
        optimizer.zero_grad()

        idx = shuffle_tensor(data)
        data = data[idx]
        target = target[idx]
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        loss_, params = optimizer.step()
        push_params_redis(params, db)
    return loss.data[0]


def train_common(epoch, args, model, data_loader, optimizer):
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                            100. * batch_idx / len(data_loader), loss.data[0]))


def test_epoch(model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(data_loader.dataset)
    return test_loss
