# NDPipe

## What is NDPipe?

NDPipe is a deep learning (DL) system designed to enhance both training and inference performance by embracing the concept of near-data processing (NDP) within storage servers. At its core, NDPipe utilizes an innovative architecture that distributes storage servers equipped with cost-effective commodity GPUs across a data center.

NDPipe is composed of two main elements: PipeStore (storage server equipped with a low-end GPU for near-data training and inference) and Tuner (training server that manages distributed PipeStores)

The original paper that introduced NDPipe is currently in the revision stage of [ACM ASPLOS 2024](https://www.asplos-conference.org/asplos2024/).

## Prerequisites

NDPipe requires hardware configurations equipped NVIDIA GPU. We detail the AWS instance types or the alternatable NVIDIA GPU-equipped machines that are suitable for NDPipe.

For running the PipeStore component, one or more instances with the following specifications are needed:

- Amazon Machine Image (AMI): Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 20.04) version 20240228.
- Instance Type: g4dn.2xlarge, which includes:
	- 8 vCPUs
	- 32 GiB Memory
	- 1 NVIDIA T4 GPU
	- 125 GiB gp3 Storage

For the Tuner component, a single instance with these specifications is required:
- Amazon Machine Image (AMI): Deep Learning Base Proprietary Nvidia Driver GPU AMI (Ubuntu 20.04) version 20240201.
- Instance Type: p3.2xlarge, featuring:
	- 8 vCPUs
	- 61 GiB Memory
	- 1 NVIDIA V100 GPU
	- 125 GiB gp3 Storage

Alternatively, NDPipe can be configured on real machines equipped with  NVIDIA GPUs.

## Installation & Execution (Fine-tuning)

### PipeStore preparation
1. Clone required repository(NDPipe) into the machine.

```
~$ git clone https://github.com/dgist-datalab/NDPipe.git
```

2. Generate an SSH key.

```
~$ ssh-keygen -t rsa
```

3. Display the public SSH key and append it to the Tuner's authorized\_keys.

```
~$ cat ~/.ssh/id_rsa.pub
```

	- On the Tuner machine, run:

	```
	~$ echo [PublicKeyContent] >> ~/.ssh/authorized_keys
	```

4. Run a Docker container with NVIDIA TensorRT.

```
~$ docker run --gpus all -it -v ~:/DataLab --name PipeStore nvcr.io/nvidia/tensorrt:20.09-py3
```

5. Set the environment variables for Tuner IP and username (replace placeholders with actual values):

```
/workspace# export TUNER_IP=[Tuner IP]
/workspace# export TUNER_USERNAME=[Tuner username]
```

6. Add the Tuner IP to known hosts for SSH:

```
/workspace# ssh-keyscan -H $TUNER_IP >> /NDPipe/.ssh/known_hosts
```

7. Update and upgrade the package lists:

```
/workspace# apt update && apt upgrade
```

8. Update pip and install required Python packages from `requirements.txt`.

```
/workspace# cd /DataLab/NDPipe/Fine_tuning/PipeStore
.../PipeStore# pip install -r requirements.txt
```

9. repare the dataset directory and download the dataset:

```
.../PipeStore# mkdir dataset
.../PipeStore# python download_dataset.py
```

10. (optional) If not using a T4 GPU, compile the model specifically for your GPU (e.g., for a different GPU):

```
trtexec --onnx=resnet50.onnx --workspace=8192 --saveEngine=resnet50.engine --buildOnly --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
```

### Tuner Preparation

1. Clone required repository(NDPipe) into the machine.

```
~$ git clone https://github.com/dgist-datalab/NDPipe.git
```

2. Update pip and install required Python packages from `requirements.txt`.

```
~$ cd NDPipe/Fine_tuning/Tuner
.../Tuner$ pip install -r requirements.txt
```

### Running the System

1. On the Tuner, start the server script with optional parameters:

```
.../Tuner$ python3.9 server.py --split_number [value] --num_of_client [value] --port [value]
```
- `--num_of_run`: Sets the pipelining strength. Default is 1.
- `--num_of_client`: The number of PipeStores. Default is 1.
- `--port`: Socket connection port. Default is 25258.

2. Simultaneously execute the main script on each PipeStore server (specifying the port if needed).

```
.../PipeStore# python main.py [port]
```
- The script uses a command-line argument for the port if provided; otherwise, it defaults to 25258.

## Installation & Execution (Offline-inference)
todo
