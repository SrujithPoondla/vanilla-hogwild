import matplotlib.pyplot as plt
import argparse
import numpy as np

# parser = argparse.ArgumentParser(description='PyTorch plot')
# parser.add_argument('--num-processes', type=int, default=2, metavar='N',
#                     help='how many training processes to use (default: 2)')
# parser.add_argument('--file', type=str, default="", metavar='N',
#                     help='Filename to plot', required = True)

# args = parser.parse_args()

# f = open('result_aws_10_3', 'r')
# g = open('result_aws_7_4', 'r')
# h = open('result_aws_11_1','r')
# h = open('result_aws_12_4', 'r')
# i = open('result_aws_14_4','r')
f = open('LeNet_11_1_1','r')
g = open('LeNet_11_1_2','r')
fig = plt.show()
title = f.readline().strip()
# title_ = h.readline()
files = [f,g]
for i,f in enumerate(files):
    subtitle = f.readline().strip()
    line = f.readline()
    if not len(line.strip()):
        break
    else:
        processes = int(line.split('=')[1].strip())
    time = []
    loss = []
    for line in f:
        if len(line.strip()) > 0:
            if 'Test' in line:
                continue
            else:
                content = line.split("\t")
                n_processes = content[0]
                if int(n_processes) == processes - 1:
                    time.append(float(content[2].split(':')[1].strip()))
                    loss.append(float(content[3].split(':')[1].strip()))
        else:
            break
    plt.ylim(2,0)
    plt.xlim(0,1000)
    # plt.xlim(max(np.array(loss)),min(np.array(loss)))
    # plt.subplot(str(len(files))+str(1)+str(i))
    plt.yticks(np.arange(0, 2, step=0.1))
    plt.xticks(np.arange(0,1000,step=50))
    fig = plt.plot(time, loss, label='N_Processes' + str(i+1))
    plt.title(title)
    plt.legend()
    plt.grid(True)
    # plt.hold()

fig = plt.show()
