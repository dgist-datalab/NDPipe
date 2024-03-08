import tensorrt as trt
import deflate, time, os, sys, gzip, gc, math
import glob
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from multiprocessing import Lock
from prefetch_generator import BackgroundGenerator, background
from tqdm import tqdm

class Model:

    def __init__(self, engine_path,
               max_workspace_size = 8192*1024*1024,
               fp16_mode = True,
               delay = 0):
        """
        TensorRT Engine Runner
        ----------------------
        engine_path: tensorrt engine
        fp16_mode  : if model has build with fp16 tag, than this parameter will be True
        delay      : emulate the delay (e.g., decompression overhead)
        """
        self.lock = Lock()
        self.delay = delay
        self.timecheck = 0
        self.TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        print("Load Cuda Engine")
        with open(engine_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        print(f"Input shape : {self.engine.get_binding_shape(0)} ({self.engine.get_binding_dtype(0)})")
        print(f"Output shape: {self.engine.get_binding_shape(1)} ({self.engine.get_binding_dtype(1)})")

        self.batch_size = self.engine.get_binding_shape(1)[0]
        self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self):
        """
        Allocate host and device buffers, and create a stream.
        Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
        And also, allocate device memory for inputs and outputs.
        """
        self.h_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=trt.nptype(self.engine.get_binding_dtype(0)))
        self.h_output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=trt.nptype(self.engine.get_binding_dtype(1)))
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        self.stream = cuda.Stream()

#Pipelining Code: Load -> Preprocessing (Decompression) -> Inference operation
    @background(max_prefetch=1)
    def load_data(self, image_list, dtype=np.float32):
        type_size = 2
        image_no = 0
        num_images = len(image_list)
        num_batch = len(image_list) // self.batch_size
        size = int(trt.volume(self.engine.get_binding_shape(0))//self.batch_size)
        for batch in range(num_batch):
            self.lock.acquire()
            if batch == num_batch-1:
                batch_size = num_images % self.batch_size
                for i in range(batch_size):
                    f = open(image_list[image_no],"rb")
                    decomp_data = deflate.gzip_decompress(f.read())
                    np.copyto(self.h_input[i*size:(i+1)*size], np.frombuffer(decomp_data, dtype=np.float32).astype('float16'))
                    image_no += 1
            else:
                batch_size = self.batch_size
                for i in range(batch_size):
                    f = open(image_list[image_no],"rb")
                    decomp_data = deflate.gzip_decompress(f.read())
                    np.copyto(self.h_input[i*size:(i+1)*size], np.frombuffer(decomp_data, dtype=np.float32).astype('float16'))
                    image_no += 1
            yield batch_size

    def do_inference(self):
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        try:
            self.lock.release()
        except:
            pass
        self.context.execute_async(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle, batch_size = 1)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

    def inference(self, data_path):
        type_size = 2
        prev_list = os.listdir(data_path)
        image_list = [data_path + item for item in prev_list]
        num_images = len(image_list)
        batch_num = math.ceil(num_images / self.batch_size)
        lddt = self.load_data(image_list)
        a = 0
        for batch_size in tqdm(lddt, total=batch_num, desc="inference progress"):
            a += 1
            start = time.perf_counter()
            self.do_inference()
            ee = time.perf_counter()
            self.timecheck += ee-start
        return num_images

if __name__ == "__main__":

    print(f"[NDPiPe] Inference Test")
    model = Model(sys.argv[1], delay=0)
    start = time.perf_counter()
    num_images = model.inference(sys.argv[2])
    end   = time.perf_counter()
    print(f"[NDPipe] inference time: {end-start}")
    print(f"[NDPipe] inference throughput : {round(num_images/(model.timecheck),2)}IPS")


