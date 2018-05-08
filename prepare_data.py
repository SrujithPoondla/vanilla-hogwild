from __future__ import print_function
import argparse
import Datasets
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                    help='Cifar-10 and MNIST are supported for now')
parser.add_argument('--n-nodes', type=int, default=1, metavar='N',
                    help='how many aws instances to start')
parser.add_argument('--node-num', type =int, default=1, metavar='N',
                    help='number of the current node')

def prepare_data(dataset, node_num):
    if dataset is 'MNIST':
        training_set = Datasets.MNIST('./mnist_data', train=True, download=True,
                             transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))]),
                             num_nodes=args.n_nodes,
                             curr_node=node_num)
        test_set = Datasets.MNIST('./mnist_data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))
        print(len(training_set), len(test_set))
    elif dataset is 'cifar10':
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
        training_set = Datasets.CIFAR10(root='./cifar10_data', train=True,
                                        download=True, transform=transform_train,
                                        num_nodes=args.n_nodes, curr_node=1)
        test_set = Datasets.CIFAR10(root='./cifar10_data', train=False,
                                   download=True, transform=transform_test)
        print(len(training_set), len(test_set))
if __name__ == '__main__':
    args = parser.parse_args()
    prepare_data(args.dataset, 1)
