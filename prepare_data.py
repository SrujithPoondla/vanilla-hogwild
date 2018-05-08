from __future__ import print_function
import argparse
import Datasets
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                    help='Cifar-10 and MNIST are supported for now')
parser.add_argument('--n-nodes', type=int, default=3, metavar='N',
                    help='how many aws instances to start')
parser.add_argument('--node-num', type=int, default=1, metavar='N',
                    help='number of the current node')


def prepare_data(dataset, node_num):
    if dataset == 'MNIST':
        training_set = Datasets.MNIST('./mnist_data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))]),
                                      num_nodes=node_num)
        test_set = Datasets.MNIST('./mnist_data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))
        print(len(training_set), len(test_set))

if __name__ == '__main__':
    args = parser.parse_args()
    prepare_data(args.dataset, args.n_nodes)
