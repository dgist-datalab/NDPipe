# NDPipe

## What is NDPipe?

<img src="./NDPipe.png" width="700" height="385">

NDPipe is a deep learning (DL) system designed to enhance both training and inference performance by embracing the concept of near-data processing (NDP) within storage servers. At its core, NDPipe utilizes an innovative architecture that distributes storage servers equipped with cost-effective commodity GPUs across a data center.

NDPipe is composed of two main elements: PipeStore (storage server equipped with a low-end GPU for near-data training and inference) and Tuner (training server that manages distributed PipeStores)

The original paper that introduced NDPipe is currently in the revision stage of [ACM ASPLOS 2024](https://www.asplos-conference.org/asplos2024/).

## Prerequisites

NDPipe requires hardware configurations equipped NVIDIA GPU. We detail the AWS instance types or the alternatable NVIDIA GPU-equipped machines that are suitable for NDPipe.

For running the PipeStore component, one or more instances with the following specifications are needed:

- Amazon Machine Image (AMI): Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 20.04) version 20240228.
- Instance Type: g4dn.2xlarge
	- 8 vCPUs
	- 32 GiB Memory
	- 1 NVIDIA T4 GPU
	- 125 GiB gp3 Storage

For the Tuner component, a single instance with these specifications is required:
- Amazon Machine Image (AMI): Deep Learning Base Proprietary Nvidia Driver GPU AMI (Ubuntu 20.04) version 20240201.
- Instance Type: p3.2xlarge
	- 8 vCPUs
	- 61 GiB Memory
	- 1 NVIDIA V100 GPU
	- 125 GiB gp3 Storage

Alternatively, NDPipe can be configured on real machines equipped with  NVIDIA GPUs.

## Installation & Execution (Fine-tuning)

### PipeStore preparation
1. Clone required repository(NDPipe) into the machine.

```
# PipeStore
~$ git clone https://github.com/dgist-datalab/NDPipe.git
```

2. Generate an SSH key.

```
# PipeStore
~$ ssh-keygen -t rsa
```

3. Display the public SSH key and append it to the Tuner's authorized\_keys.

```
# PipeStore
~$ cat ~/.ssh/id_rsa.pub
```

- On the Tuner machine, run:

	```
	# Tuner
	~$ echo [PublicKeyContent] >> ~/.ssh/authorized_keys
	```

4. Run a Docker container with NVIDIA TensorRT.

```
# PipeStore
~$ docker run --gpus all -it -v ~:/DataLab --name PipeStore nvcr.io/nvidia/tensorrt:20.09-py3
```

5. Set the environment variables for Tuner IP and username (replace placeholders with actual values):

```
# PipeStore
/workspace# export TUNER_IP=[Tuner IP]
/workspace# export TUNER_USERNAME=[Tuner username]
```

6. Add the Tuner IP to known hosts for SSH:

```
# PipeStore
/workspace# ssh-keyscan -H $TUNER_IP >> /DataLab/.ssh/known_hosts
```

7. Update and upgrade the package lists:

```
# PipeStore
/workspace# apt update && apt upgrade
```

8. Update pip and install required Python packages from `requirements.txt`.

```
# PipeStore
/workspace# cd /DataLab/NDPipe/Fine_tuning/PipeStore
.../PipeStore# pip install -r requirements.txt
```

9. repare the dataset directory and download the dataset:

```
# PipeStore
.../PipeStore# mkdir dataset
.../PipeStore# python download_dataset.py
```

10. (optional) If not using a T4 GPU, compile the model specifically for your GPU (e.g., for a different GPU):

```
# PipeStore
trtexec --onnx=resnet50.onnx --workspace=8192 --saveEngine=resnet50.engine --buildOnly --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
```

### Tuner Preparation

1. Clone required repository(NDPipe) into the machine.

```
# Tuner
~$ git clone https://github.com/dgist-datalab/NDPipe.git
```

2. Update pip and install required Python packages from `requirements.txt`.

```
# Tuner
~$ cd NDPipe/Fine_tuning/Tuner
.../Tuner$ pip install -r requirements.txt
```

### Running the System (< 2mins to run)

1. On the Tuner, start the server script with optional parameters:

```
# Tuner
.../Tuner$ python3.9 server.py --num_of_run [value] --num_of_client [value] --port [value]
```
- `--num_of_run` or	`-r`: Sets the pipelining strength. Default is 1.
- `--num_of_client` or `-n`: The number of PipeStores. Default is 1.
- `--port` or `p`: Socket connection port. Default is 25258.

2. Simultaneously execute the main script on each PipeStore server (specifying the port if needed).

```
# PipeStore
.../PipeStore# python main.py [port]
```
- The script uses a command-line argument for the port if provided; otherwise, it defaults to 25258.

## Installation & Execution (Offline-inference)
todo
