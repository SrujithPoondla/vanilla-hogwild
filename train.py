import os
import timeit
import time
import pymp
import torch
import torch.nn.functional as F
import redis
from torch.autograd import Variable
from torchvision import datasets, transforms
from sgd import SGD
from rediscluster import StrictRedisCluster
from torch.optim import sgd

from common_functions import push_params_redis, get_shapes, get_params_redis, set_params, push_params_redis_init, \
    check_param_exists


def train(args, model):

    db = []
    if args.is_redis:
        startup_nodes = []
        for node in args.hosts.split(' '):
            startup_nodes.append({'host': str(node.split(':')[0]), "port": "6379"})
        if len(startup_nodes) > 2:
            db = StrictRedisCluster(startup_nodes=startup_nodes, decode_responses=True)
        else:
            db = redis.ConnectionPool(host='localhost', port=6379, db=0)
            db = redis.StrictRedis(connection_pool=db)
        # startup_nodes = [{"host": "127.0.0.1", "port": "30001"},{"host": "127.0.0.1", "port": "30002"},{"host": "127.0.0.1", "port": "30003"}]
        # db = StrictRedisCluster(startup_nodes=startup_nodes, decode_responses=True)

        params_exists = check_param_exists(model, db)
        if not params_exists:
            push_params_redis_init(model,db)

    shapes = get_shapes(model)

    # Print total number of processes
    print('Num process in each node: '+ str(args.num_processes))

    # Using pymp to parallelise the training
    epoch_start_time = 0
    with pymp.Parallel(args.num_processes) as p:

        if args.dataset == "MNIST":
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
        elif args.dataset == "cifar10":
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            # load training and test set here:
            training_set = datasets.CIFAR10(root='./cifar10_data', train=True,
                                            download=True, transform=transform_train)
            train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                                       shuffle=True)
            testset = datasets.CIFAR10(root='./cifar10_data', train=False,
                                       download=True, transform=transform_test)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                                      shuffle=False)


        # Optimizer declaration
        if args.is_redis:
            optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        else:
            optimizer = sgd.SGD(model.parameters(), lr = args.lr, momentum= args.momentum)

        if not p.thread_num:
            time.sleep(20)

        epoch_start_time = timeit.default_timer()
        for epoch in range(args.epochs):
            train_process(p.thread_num, optimizer, train_loader, model, args, shapes, db, epoch, epoch_start_time)
            # if p.thread_num :
            # print('PID{}\tTrain Epoch: {}\t time: {} \tLoss: {:.6f}'.format(p.thread_num, epoch, epoch_time, loss))
            test_epoch(model, test_loader)
            # print('PID{}\tTrain Epoch: {}\t time: {} \tLoss: {:.6f}'.format(p.thread_num, epoch, epoch_time, test_loss))


def shuffle_tensor(tensor):
    return torch.randperm(tensor.size(0)).long()


def train_process(thread_num, optimizer, train_loader, model, args, shapes, db, epoch, epoch_start_time):
    # epoch_start_time = timeit.default_timer()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.is_redis:
            set_params(optimizer, get_params_redis(db, shapes))
        model.train()
        optimizer.zero_grad()
        idx = shuffle_tensor(data)
        data = data[idx]
        target = target[idx]
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        if args.is_redis:
            loss_, params = optimizer.step()
            push_params_redis(params, db)
        else:
            optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\ttime: {}\tLoss: {:.6f}'.format(
                thread_num, epoch, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), timeit.default_timer()-epoch_start_time, loss.item()))

    # return loss.data[0]


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
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
