import torch
print(f"Device capability: {torch.cuda.get_device_capability()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
# List all files being used for ROCm to see if it's hitting the wrong path
import os
print(os.environ.get('LD_LIBRARY_PATH'))