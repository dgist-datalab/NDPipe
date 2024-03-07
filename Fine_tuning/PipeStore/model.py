import os, paramiko
from tqdm import tqdm
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time
import math
import cv2
from multiprocessing import Lock
from prefetch_generator import BackgroundGenerator, background

class Model:
    def __init__(self,
                engine_path,
                ssd_num,
                max_workspace_size = 8192*1024*1024,
                max_batch_size = 128,
                fp16_mode = True,
                in_dtype = trt.float16,
                out_dtype = trt.float16,
                evaluate_mode = False,
                classifier_path=None,
                num_classes=None,
                feature_dim = 224,
                server_dir = '/'):
        self.feature_dim = feature_dim
        self.max_batch_size = max_batch_size
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.scp_c = 0
        self.evaluate_mode = evaluate_mode
        self.num_classes = num_classes
        self.offset = 0
        self.start_pos = 0
        self.TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        self.server_dir = server_dir
        self.ssh = paramiko.SSHClient()
        self.ssh.load_host_keys(os.path.expanduser(os.path.join("../../..", ".ssh", "known_hosts")))
        key = paramiko.RSAKey.from_private_key_file(os.path.join("../../..", ".ssh", "id_rsa"))
        self.ssh.connect(os.environ['TUNER_IP'], username=os.environ['TUNER_USERNAME'], pkey=key)
        print("Load Cuda Engine")
        with open(engine_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.allocate_buffers()
        self.context = self.engine.create_execution_context()
        self.ssd_num = ssd_num


    def allocate_buffers(self):
        # Allocate host and device buffers, and create a stream.
        # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
        self.h_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=trt.nptype(self.in_dtype))
        self.h_output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=trt.nptype(self.out_dtype))
        # Allocate device memory for inputs and outputs.
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
        self.stream = cuda.Stream()

    @background(max_prefetch=3)
    def load_data(self, data_path, num_images):
        if self.in_dtype == trt.float16:
            type_size = 2
        else:
            type_size = 4
        num_batch = math.ceil(num_images / self.max_batch_size)
        with open(data_path, "rb") as frp:
            frp.seek(self.start_pos*(self.feature_dim*self.feature_dim*3)*type_size)
            for batch in range(num_batch):
                if batch == math.ceil((self.offset)/self.max_batch_size)-1:
                    size = self.feature_dim*self.feature_dim*3*((self.offset) % self.max_batch_size)
                    batch_size = (self.offset) % self.max_batch_size
                    np.copyto(self.h_input[:size], np.frombuffer(frp.read(self.feature_dim*self.feature_dim*3*type_size*batch_size), dtype=trt.nptype(self.in_dtype)))
                    yield batch_size
                    break
                if batch == num_batch-1:
                    size = self.feature_dim*self.feature_dim*3*(num_images % self.max_batch_size)
                    batch_size = num_images % self.max_batch_size
                    np.copyto(self.h_input[:size], np.frombuffer(frp.read(self.feature_dim*self.feature_dim*3*type_size*batch_size), dtype=trt.nptype(self.in_dtype)))
                else:
                    size = self.feature_dim*self.feature_dim*3*self.max_batch_size
                    batch_size = self.max_batch_size
                    np.copyto(self.h_input, np.frombuffer(frp.read(self.feature_dim*self.feature_dim*3*type_size*self.max_batch_size), dtype=trt.nptype(self.in_dtype)))
                yield batch_size

    def do_inference(self):
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        # Run inference.
        self.context.execute_async(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle, batch_size = 1)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        # Synchronize the stream
        self.stream.synchronize()

    def do_scp(self, path):
        sftp = self.ssh.open_sftp()
        sftp.put(path, self.server_dir+'/'+path)
        sftp.close()


    def write_label(self, source_path, final_path, offset, length, IsExtra):
        if IsExtra == True:
            mode = "ab"
        else:
            mode = "wb"

        with open(final_path, mode) as fwp:
            with open(source_path, "rb") as frp:
                frp.seek(offset*4)
                fwp.write(frp.read(length*4))

    def extract_features(self, data_path, label_path, feature_path, feature_label_path, client, append=False, IsExtra=False, extra_num=0, split=1):
        if append == False:
            mode = "wb"
        else:
            mode = "ab"
        inference = 0
        count = 0
        num_images = os.path.getsize(data_path) // (self.feature_dim*self.feature_dim*3*2)

        sub_loop = 0
        sub_feature = (num_images/split) - (num_images/split) % self.max_batch_size
        last_sub_feature = num_images - sub_feature*split


        if IsExtra == True:
            self.start_pos = num_images - extra_num
            num_images = extra_num


        features = np.zeros([num_images,2048], dtype=trt.nptype(self.out_dtype))
        batch_num = math.ceil(num_images / self.max_batch_size)
        tcp_num = int(batch_num*0.3)
        loop_count = 0
        throughput = 0
        balance_time = 0
        flag = 0
        sub_count = 0
        LB_mode = 0
        start = time.perf_counter()
        feature_path_list = []
        label_path_list = []
        fopen_list = []
        if IsExtra == False:
            for i in range(split):
                sub_feature_path = f"train_feature_{self.ssd_num}_"+str(i)+".dat"
                sub_label_path = f"train_label_{self.ssd_num}_"+str(i)+".dat"
                label_path_list.append(sub_label_path)
                feature_path_list.append(sub_feature_path)
                sub_f = open(sub_feature_path, mode)
                fopen_list.append(sub_f)
        else:
            sub_feature_path = feature_path
            sub_label_path = feature_label_path
            label_path_list.append(sub_label_path)
            feature_path_list.append(sub_feature_path)
            sub_f = open(sub_feature_path, mode)
            fopen_list.append(sub_f)
        
        lddt = self.load_data(data_path, num_images)

        for batch_size in tqdm(lddt, total=batch_num):
            s = time.perf_counter()
            t1 = time.perf_counter()

            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            self.context.execute_async(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle, batch_size = 1)
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()
            t2 = time.perf_counter()
            sub_count += batch_size
            count += batch_size
            fopen_list[sub_loop].write(self.h_output[:batch_size*self.engine.get_binding_shape(1)[1]].tobytes())
            loop_count += 1
            inference += t2 - t1
            throughput = count/inference
            e = time.perf_counter()
            balance_time += (e-s)
            if sub_count == sub_feature and sub_loop != split-1:
                self.do_scp(feature_path_list[sub_loop])
                self.write_label(label_path, label_path_list[sub_loop], count-sub_count, sub_count, IsExtra)
                self.do_scp(label_path_list[sub_loop])
                sub_count = 0
                sub_loop += 1

        if IsExtra == False:
            if extra_num == 0:
                self.do_scp(feature_path_list[-1])
                self.write_label(label_path, label_path_list[-1], count-sub_count, sub_count, IsExtra)
                self.do_scp(label_path_list[-1])
            else:
                self.write_label(label_path, label_path_list[-1], count-sub_count, sub_count, IsExtra)
        elif IsExtra == True:
            self.do_scp(feature_path_list[-1])
            self.write_label(label_path, label_path_list[-1], count-sub_count, sub_count, IsExtra)
            self.do_scp(label_path_list[-1])

        end = time.perf_counter()
        total = end - start
        re_num = self.offset
        self.offset = 0
        self.start_pos = 0
        extra_path, extra_label_path = None, None
        if IsExtra == False:
            return inference, total, count, extra_num, extra_path, re_num, extra_label_path, feature_path_list[-1], label_path_list[-1]
        else:
            return inference, total, count

    def evaluate(self, data_path, label_path):
        inference = 0
        count = 0
        top_1 = 0
        top_5 = 0
        num_images = os.path.getsize(data_path) // (self.feature_dim*self.feature_dim*3*2)

        features = np.zeros([num_images,trt.volume(self.engine.get_binding_shape(1))], dtype=trt.nptype(self.dtype))
        batch_num = math.ceil(num_images / self.max_batch_size)

        start = time.perf_counter()

        for batch_size in self.load_data(data_path, num_images):
            # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
            # probability that the image corresponds to that label
            t1 = time.perf_counter()
            self.do_inference()
            t2 = time.perf_counter()

            # We use the highest probability as our prediction. Its index corresponds to the predicted label.
            features[count:count+batch_size,:] = self.h_output[:batch_size*trt.volume(self.engine.get_binding_shape(1))].reshape(batch_size, trt.volume(self.engine.get_binding_shape(1)))
            count += batch_size

            inference += t2 - t1
        end = time.perf_counter()
        total = end - start

        with open(label_path, "rb") as f:
            label = np.frombuffer(f.read(), np.int64)

        for i in range(count):
            pred = np.flip(np.argsort(features[i,:])[-5:])
            if label[i] == pred[0]:
                top_1 += 1
            if label[i] in pred:
                top_5 += 1
            # if i < 50:
            #     print(i, label[i], pred)

        print("Number of Images", count, "Top_1 accuracy:", top_1, top_1/count, "Top_5 accuracy:", top_5, top_5/count)
        print("Inference:", inference, count/inference)
        print("Total:", total, count/total)
        return inference, total, count, top_1, top_5

