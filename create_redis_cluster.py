import argparse
import subprocess
import boto3
import os
import sys

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--num-instances', type=int, default=1, metavar='N',
                    help='Number of instances (default 3)')
parser.add_argument('--instance-type', type=str, default="m5.2xlarge", metavar='N')


class Cfg(dict):

   def __getitem__(self, item):
       item = dict.__getitem__(self, item)
       if type(item) == type([]):
           return [x % self if type(x) == type("") else x for x in item]
       if type(item) == type(""):
           return item % self
       return item

cfg = Cfg({
    "name" : "Timeout",      # Unique name for this specific configuration
    "key_name": "aws-key-spot-instance",          # Necessary to ssh into created instances
    "num_nodes" : 3,
    "method" : "spot",
    # Region speficiation
    "region" : "us-east-1a",
    "availability_zone" : "us-east-1a",
    # Machine type - instance type configuration.
    "instance_type" : "m4.2xlarge",
    # please only use this AMI for pytorch
    "image_id": "ami-f60aba8e",
    # Launch specifications
    "spot_price" : 0.15,                 # Has to be a string
    # SSH configuration
    "ssh_username" : "ubuntu",            # For sshing. E.G: ssh ssh_username@hostname
    "path_to_keyfile" : "~/.ssh/",


    # Command specification
    # Master pre commands are run only by the master
    "master_pre_commands" :
    [
        "cd my_mxnet",
        "git fetch && git reset --hard origin/master",
        "cd cifar10",
        "ls",
        # "cd distributed_tensorflow/DistributedResNet",
        # "git fetch && git reset --hard origin/master",
    ],
    # Pre commands are run on every machine before the actual training.
    "pre_commands" :
    [
        "cd my_mxnet",
        "git fetch && git reset --hard origin/master",
        "cd cifar10",
    ],
    # Model configuration
    "batch_size" : "32",
    "max_steps" : "2000",
    "initial_learning_rate" : ".001",
    "learning_rate_decay_factor" : ".95",
    "num_epochs_per_decay" : "1.0",
    # Train command specifies how the ps/workers execute tensorflow.
    # PS_HOSTS - special string replaced with actual list of ps hosts.
    # TASK_ID - special string replaced with actual task index.
    # JOB_NAME - special string replaced with actual job name.
    # WORKER_HOSTS - special string replaced with actual list of worker hosts
    # ROLE_ID - special string replaced with machine's identity (E.G: master, worker0, worker1, ps, etc)
    # %(...)s - Inserts self referential string value.
    "train_commands" :
    [
        "echo ========= Start ==========="
    ],
})


args = parser.parse_args()
# Create 3 instances
client = boto3.client("ec2", region_name=configuration["region"])
ec2 = boto3.resource("ec2", region_name=configuration["region"])

prices=client.describe_spot_price_history(InstanceTypes=[args.instance_type],MaxResults=1,ProductDescriptions=['Linux/UNIX (Amazon VPC)'],AvailabilityZone='us-east-1a')
price = prices['SpotPriceHistory'][0]['SpotPrice']
print(price)


def create_cluster(configuration):

    # Launch instances as specified in the configuration.
    def launch_instances():
       method = "spot"
       if "method" in configuration.keys():
          method = configuration["method"]
       # worker_instance_type, worker_count = configuration["worker_type"], configuration["n_workers"]
       # master_instance_type, master_count = configuration["master_type"], configuration["n_masters"]
       instance_type, instance_count = configuration["isntance_type"], configuration["n_node"]
       specs = [(worker_instance_type, worker_count),
                (master_instance_type, master_count)]
       for (instance_type, count) in specs:
          launch_specs = {"KeyName" : configuration["key_name"],
                          "ImageId" : configuration["image_id"],
                          "InstanceType" : instance_type,
                          "Placement" : {"AvailabilityZone":configuration["availability_zone"]},
                          "SecurityGroups": ["default"]}
          if method == "spot":
             # TODO: EBS optimized? (Will incur extra hourly cost)
             client.request_spot_instances(InstanceCount=count,
                                           LaunchSpecification=launch_specs,
                                           SpotPrice=configuration["spot_price"])
          elif method == "reserved":
             client.run_instances(ImageId=launch_specs["ImageId"],
                                  MinCount=count,
                                  MaxCount=count,
                                  KeyName=launch_specs["KeyName"],
                                  InstanceType=launch_specs["InstanceType"],
                                  Placement=launch_specs["Placement"],
                                  SecurityGroups=launch_specs["SecurityGroups"])
          else:
             print("Unknown method: %s" % method)
             sys.exit(-1)

if __name__ == '__main__':
    create_cluster(cfg)