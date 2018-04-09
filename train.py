import os
import timeit

import pymp
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch import multiprocessing as mp


def train(args, model):

    # torch.manual_seed(args.seed + rank)
    # dataset = datasets.MNIST('../data', train=True, download=True,
    #                 transform=transforms.Compose([
    #                     transforms.ToTensor(),
    #                     transforms.Normalize((0.1307,), (0.3081,))
    #                 ]))
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
    processes = []
    data =[]
    for batch, (train_data, target_data) in enumerate(train_loader):
        data.append((train_data, target_data))

    epoch_time = 0
    with pymp.Parallel(args.num_processes) as p:
        for epoch in range(args.epochs):
            start_time = timeit.default_timer()
            train_data,target_data = data[p.thread_num]
            idx = shuffle_tensor(train_data)
            train_data = train_data.index(idx)
            target_data = target_data.index(idx)
            train_process(p.thread_num,train_data,target_data,model,args)
                    # processes.append(p)
                # for p in processes:
                #     p.join()
            epoch_time = epoch_time + timeit.default_timer()-start_time
            if p.thread_num is 0:
                print('PID{}\tTrain Epoch: {}\t time: {} \tLoss: {:.6f}'.format(os.getpid(),
                    epoch, epoch_time, test_epoch(model, test_loader)))


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


def train_process(batch,train_data,target_data,model,args):
    pid = os.getpid()
   # print(str(pid)+"started")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    model.train()
    data, target = Variable(train_data), Variable(target_data)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()


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
