from __future__ import print_function
import argparse
import torch
from diff_models import *

from resnet import ResNet18

from train import train

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=1, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--nnet-arch', type=str, default='LeNet', metavar='N',
                    help='LeNet and ResNet are supported for now')
parser.add_argument('--hosts', type=str, default='127.0.0.1', metavar='N',
                    help='Defaults to loopback address')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                    help='Cifar-10 and MNIST are supported for now')
parser.add_argument('--is-redis', type=bool, default=True, metavar='N',
                    help='Cifar-10 and MNIST are supported for now')


def build_model(model_name, dataset):
    # build network
    if model_name == "LeNet":
        if dataset == "MNIST":
            return LeNetMnist();
        elif dataset == "cifar10":
            return LeNetCifar10()
    elif model_name == "ResNet":
        return ResNet18()
    # elif model_name == "ResNet34":
    #     return ResNet34()
    # elif model_name == "ResNet50":
    #     return ResNet50()

if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    print(args.nnet_arch,args.dataset)
    model = build_model(args.nnet_arch, args.dataset)
    # model = models.resnet18()
    # for i in list(model.parameters()): print(i.shape)
    #    model.share_memory() # gradients are allocated lazily, so they are not shared here
    train(args, model)

