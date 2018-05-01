import argparse
import os
import subprocess
import boto3
import paramiko as pm
import time

parser = argparse.ArgumentParser(description='Asynchronous SGD updates using Redis')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--nnet-arch', type=str, default='LeNet', metavar='N',
                    help='LeNet and ResNet are supported for now')
parser.add_argument('--n-nodes', type=int, default=1, metavar='N',
                    help='how many aws instances to start')
parser.add_argument('--dataset', type=str, default='Cifar-10', metavar='N',
                    help='Cifar-10 and Mnist are supported for now')
parser.add_argument('--is-redis',type=bool,default=True, metavar='N',
                    help="Choose whether the model to be trained using redis or not."
                         " If not using Redis, model will be trained on single process")


args = parser.parse_args()


def connect_to_instance(host):
    client = pm.SSHClient()
    host = pub_ip
    client.set_missing_host_key_policy(pm.AutoAddPolicy())
    client.connect(host, username='ubuntu', key_filename='/Users/srujithpoondla/.ssh/aws-key-spot-instance')
    return client

curr_pwd = os.getcwd()
os.chdir('/Users/srujithpoondla/largescaleml_project/aws-setup/')
for proc in range(args.n_nodes):
    print(os.getcwd())
    subprocess.check_call(['/Users/srujithpoondla/largescaleml_project/aws-setup/request-spot-instance.sh', 'm5.2xlarge', '0.1503', 'vpc-1b056b60'])
ec2 = boto3.resource('ec2')
filters = [
    {
        'Name': 'instance-state-name',
        'Values': ['running']
    }
]
# filter the instances based on filters() above
instances = ec2.instances.filter(Filters=filters)

# instantiate empty array
public_ip_list = []
private_ip_list = []
for instance in instances:
    # for each instance, append to array
    private_ip_list.append(instance.private_ip_address)
    public_ip_list.append(instance.public_ip_address)

status = True
statuses = 0
while status:
    for stat in ec2.meta.client.describe_instance_status()['InstanceStatuses']:
        if stat['InstanceStatus']['Status'] is not 'ok':
            time.sleep(10)
            continue
        else:
            statuses = statuses+1
    if statuses == len(ec2.meta.client.describe_instance_status()['InstanceStatuses']):
        status = False

# ssh in to each instance and set up the code base and redis
os.chdir('/Users/srujithpoondla/aws_scripts/')
private_ip_str = ''
for pub_ip, priv_ip in public_ip_list, private_ip_list:
    private_ip_str = private_ip_str+" "+str(priv_ip)+":6379"
    client = connect_to_instance(pub_ip)
    stdin, stdout, stderr = client.exec_command('source activate largescale')
    if 'error' in stdout.read():
        print("Error activating conda environment")
        break
    if args.n_nodes < 3:
        redis_str = 'redis-server'
    else:
        redis_str = 'redis-server redis-stable/redis.conf'
    stdin, stdout, stderr = client.exec_command(redis_str)
    if 'error' in stdout.read():
        print('Error starting redis')
        break

train_cmd_str = 'python3 ./async_sgd_redis/main.py'
train_cmd_str = train_cmd_str + '--num-processes=' + str(args.num_processes) \
                        + '--epochs=' + str(args.epochs) \
                        + '--lr=' + str(args.lr)\
                        + '--arch=' + args.nnet_arch\
                        + '--dataset=' + args.dataset\
                        + '--batch-size=' + str(args.batch_size)\
                        + '--hosts='+private_ip_str\
                        + '&'

if args.n_nodes >= 3:
    connect_to_instance(public_ip_list[0])
    stdin, stdout, stderr = client.exec_command('./redis-stable/src/redis-trib.rb create '+private_ip_str)
    if 'success' not in stdout.read():
        print('Cluster creation failed')

pid = []
for ip, instance in public_ip_list, instances:
    client = connect_to_instance(ip)
    stdin, stdout, stderr = client.exec_command(train_cmd_str)
    pid.append(stdout.split(' ')[1])


