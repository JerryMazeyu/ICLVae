import torch
import time
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            del obj
    print("Clear GPU OK!")


def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free >= min_memory_available:
            break
        print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        raise RuntimeError(f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")

def get_gpu_memory(device_id=0):
    total_mem = torch.cuda.get_device_properties(device_id).total_memory
    available_mem = total_mem - torch.cuda.memory_allocated(device_id)
    print(f"Total Memory: {total_mem / (1024 ** 3):.2f} GB")
    print(f"Available Memory: {available_mem / (1024 ** 3):.2f} GB")

