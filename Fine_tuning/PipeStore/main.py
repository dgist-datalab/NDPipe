import warnings
warnings.filterwarnings('ignore')

from model import Model
import socket
import os
import sys
import glob
import time

arg_port = sys.argv[1:]

# dataset list load
train_image_list = glob.glob("./dataset/train_image.dat")
train_image_list.sort()
train_label_list = glob.glob("./dataset/train_label.dat")
train_label_list.sort()
validation_image_list = glob.glob("./dataset/test_image.dat")
validation_image_list.sort()
validation_label_list = glob.glob("./dataset/test_label.dat")
validation_label_list.sort()

train_image_list = train_image_list[:]
train_label_list = train_label_list[:]
validation_image_list = validation_image_list[:]
validation_label_list = validation_label_list[:]

os.system("rm -rf train_feature_*.dat")
os.system("rm -rf train_label_*.dat")
os.system("rm -rf test_feature_*.dat")
os.system("rm -rf test_label_*.dat")

# server setting
SERVER = os.environ['TUNER_IP']

if arg_port:
	PORT = int(arg_port[0])
else:
	PORT = 25258

start_message = '''===================================================
                     PipeStore
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

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

while True:
    try:
        client.connect((SERVER, PORT))
        break
    except:
        pass

SALS_value = int(client.recv(4096).decode())
get_here = f'dir:{os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))}'
client.sendall(get_here.encode())
server_dir, ssd_num = client.recv(4096).decode().split()
client.sendall("Feature Extraction Started".encode())
data = client.recv(4096)
print("Receive:", data.decode())

train_inference = 0
train_total = 0
train_count = 0

val_inference = 0
val_total = 0
val_count = 0

start = time.perf_counter()

re_mode = False
ex_mode = False

extractor = Model(engine_path="resnet50.engine", ssd_num=ssd_num, feature_dim = 224, server_dir=server_dir[4:])
for idx, path in enumerate(validation_image_list):
    class_inference, class_total, class_count = extractor.extract_features(path, validation_label_list[idx], f"test_feature_{ssd_num}.dat", f"test_label_{ssd_num}.dat", client, append=True, IsExtra=True, extra_num = 10000)
    val_inference += class_inference
    val_total += class_total
    val_count += class_count

for idx, path in enumerate(train_image_list):
    class_inference, class_total, class_count, extra_num, extra_path, re_num, extra_label_path, last_feature_path, last_label_path = extractor.extract_features(path, train_label_list[idx], "None", "None", client, append=True, IsExtra=False, extra_num = 0, split=SALS_value)
    train_inference += class_inference
    train_total += class_total
    train_count += class_count

end = time.perf_counter()

inference = train_inference+val_inference
total = train_total+val_total
count = train_count + val_count
extract_time = str(total)
client.sendall(extract_time.encode())
client.close()
extractor.ssh.close()
del extractor

