from classifier import Classifier
from utils.load_module import *
import time, os
import argparse

os.system("rm -rf *.dat")

### Get system parameters
parser = argparse.ArgumentParser()
parser.add_argument('--num_of_run', '-r', type=int, default=1, help='Set the pipelining strength\nDefault value is 1')
parser.add_argument('--num_of_client', '-n', type=int, default=1, help='The number of SOFA SSDs\nDefault value is 1')
parser.add_argument('--port', '-p', type=int, default=25258, help='Set the socket connection port\nDefault value is 25258')
args = parser.parse_args()

start_message = '''===================================================
                       Tuner
###################################################
#  #######  ##       ###       ####################
#   ######  ##  ####  ##  ####  ###################
#    #####  ##  ####  ##  ####  ###################
#  ##  ###  ##  ####  ##  ####  ##  ##    ####   ##
#  ###  ##  ##  ####  ##       #######  #  ##  #  #
#  ####  #  ##  ####  ##  ########  ##  #  ##     #
#  #####    ##  ####  ##  ########  ##    ###  ####
#  ######   ##  ####  ##  ########  ##  #####  #  #
#  #######  ##       ###  ########  ##  ######   ##
###################################################
===================================================
                    ASPLOS 2024
---------------------------------------------------'''
print(start_message)

overall_start = time.perf_counter()

### Set communication to clients(SOFA SSDs)
comm = CommUnit(args.split_number, args.num_of_client, args.port, client=CLIENT)
comm.get_SSD_path()
comm.send_message(f'dir:{os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))}', True)
comm.get_message("Feature Extraction Started")

train_feature_path = []
train_label_path = []
test_feature_path = []
test_label_path = []

for i in range(comm.split_number):
    for j in range(comm.number_of_client):
        if i == 0:
            test_feature_path.append(f'test_feature_{j}.dat')
            test_label_path.append(f'test_label_{j}.dat')
        train_feature_path.append(f'train_feature_{j}_{i}.dat')
        train_label_path.append(f'train_label_{j}_{i}.dat')

total_image_path = train_feature_path + test_feature_path
model_path = None

for i in range(comm.split_number):
    print(f'Run {i+1} Start')
    flag = 0
    for j in range(comm.number_of_client):
        past_size = -1
        ### Check ending of feature extraction
        while True:
            if os.path.isfile(train_feature_path[i*comm.number_of_client+j]) and os.path.isfile(train_label_path[i*comm.number_of_client+j]):
                current_size = os.path.getsize(train_feature_path[i*comm.number_of_client+j])
                if i == 0 and flag == 0:
                    time.sleep(0.5)
                    flag = 1
                if current_size == past_size:
                    break
                else:
                    past_size = current_size
                    time.sleep(0.5)

    ### Classifier start
    clsfier = Classifier(num_classes=1000, feature_dim=2048,
                        train_feature_path=train_feature_path[i*comm.number_of_client:(i+1)*comm.number_of_client],
                        train_label_path=train_label_path[i*comm.number_of_client:(i+1)*comm.number_of_client],
                        test_feature_path=test_feature_path, test_label_path=test_label_path, lb = comm.split_number, loop = i)
    save_path, accuracy = clsfier.train(model_path)
    model_path = save_path

overall_end = time.perf_counter()

extract_time_list = []
for i in range(comm.number_of_client):
    extract_time_list.append(float((comm.client[i].recv(4096)).decode()))

num_images = 0
for image_path in total_image_path:
    num_images += os.path.getsize(image_path)/2048/2

# result print
print('=====================================================================')
print('NDPipe Experiment Information')
print('---------------------------------------------------------------------')
print("Feature extraction time (sec):              ", max(extract_time_list))
print("Feature extraction throughput (image/sec):  ", 1/((max(extract_time_list))/num_images))
print("Overall fine-tuning time (sec):             ", overall_end-overall_start)
print('=====================================================================')
