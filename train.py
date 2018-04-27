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
from redis import StrictRedis
from rediscluster import StrictRedisCluster

from common_functions import push_params_redis, get_shapes, get_params_redis, set_params, push_params_redis_init


def train(args, model):
    # startup_nodes = [{"host": "127.0.0.1", "port": "30001"},{"host": "127.0.0.1", "port": "30002"}]
    db = redis.ConnectionPool(host='localhost', port=6379, db=0)
    db = redis.StrictRedis(connection_pool=db)
    # db = StrictRedisCluster(startup_nodes=startup_nodes, decode_responses=True)

    start_time = timeit.default_timer()
    push_time = timeit.default_timer()
    push_params_redis_init(model,db)
    # push_params_redis(model, db)
    # print("Time to get model from redis: "+str(timeit.default_timer()-push_time))
    shapes = get_shapes(model)


    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        num_processes=args.num_processes, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        num_processes=args.num_processes, shuffle=True, num_workers=1)

    # Dividing data to features and labels
    data =[]
    for batch, (train_data, target_data) in enumerate(train_loader):
        data.append((train_data, target_data))

    # Print total number of processes
    print(args.num_processes)

    # Using pymp to parallelise the training
    epoch_time = 0
    with pymp.Parallel(args.num_processes) as p:
        for epoch in range(args.epochs):
            epoch_start_time = timeit.default_timer()
            train_data,target_data = data[p.thread_num]
            shuffle_time = timeit.default_timer()
            idx = shuffle_tensor(train_data)
            train_data = train_data.index(idx)
            target_data = target_data.index(idx)
            # print("Time to shuffle data: "+str(timeit.default_timer()-shuffle_time))
            loss = train_process(p.thread_num,train_data,target_data,model,args,shapes,db)
                    # processes.append(p)
                # for p in processes:
                #     p.join()
            epoch_time = epoch_time+ timeit.default_timer() - epoch_start_time
            if p.thread_num is 1:
                print('PID{}\tTrain Epoch: {}\t time: {} \tLoss: {:.6f}'.format(p.thread_num, epoch, epoch_time, loss))


# def shuffle_tensor (tensor):
# 	shuffle_indexes = torch.randperm(tensor.size(1))
# 	tensor_shuffled = torch.FloatTensor(tensor.size())
# 	for i=1,tensorsize(1),1 do
# 		tensor_shuffled[i] = tensor[shuffle_indexes[i]]
# 	end
# 	return tensor_shuffled
# end

def shuffle_tensor(tensor):
    return torch.randperm(tensor.size(0)).long()


def train_process(thread_num,train_data,target_data,model,args,shapes,db):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    model.train()
    data, target = Variable(train_data), Variable(target_data)

    get_params_time = timeit.default_timer()
    set_params(optimizer, get_params_redis(db, shapes))
    optimizer.zero_grad()

    # print("Time to get params from redis in thread {}: "+ str(timeit.default_timer()-get_params_time), thread_num)

    forward_time = timeit.default_timer()
    output = model(data)
    # print("Time to do forward pass in thread {}: "+ str(timeit.default_timer()-forward_time),thread_num)

    loss = F.nll_loss(output, target)
    backward_time = timeit.default_timer()
    loss.backward()
    # print("Time for backward pass in thread {}: "+str(timeit.default_timer()-backward_time),thread_num)

    # optimizer.step()
    push_params_time = timeit.default_timer()
    push_params_redis(optimizer, db)
    # print("Time to push params to redis in thread {}:"+ str(timeit.default_timer()-push_params_time), thread_num)
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
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(data_loader.dataset)
    return test_loss
