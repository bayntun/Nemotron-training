import importlib
import os

import torch

print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("LD_LIBRARY_PATH", os.getenv("LD_LIBRARY_PATH", ""))
print("CUDA_HOME", os.getenv("CUDA_HOME", ""))

for name in ["mamba_ssm", "bitsandbytes", "deepspeed"]:
    try:
        mod = importlib.import_module(name)
        print(name, "OK", getattr(mod, "__version__", "?"), getattr(mod, "__file__", "?"))
    except Exception as e:
        print(name, "ERR", repr(e))
