import torch
from torch.utils.cpp_extension import CUDA_HOME
import os

print("=== CUDA Debug Information ===")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"CUDA_HOME: {CUDA_HOME}")
print(f"CUDA_HOME exists: {CUDA_HOME is not None and os.path.exists(CUDA_HOME) if CUDA_HOME else False}")

# Check environment variables
print(f"Environment CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"Environment PATH: {os.environ.get('PATH', '')}")

# Check if nvcc is accessible
import subprocess
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    print(f"nvcc accessible: {result.returncode == 0}")
    if result.returncode == 0:
        print(f"nvcc version: {result.stdout.strip()}")
except:
    print("nvcc not accessible")

# Check what would be compiled
if torch.cuda.is_available() and CUDA_HOME is not None:
    print("✓ CUDA compilation will be enabled")
else:
    print("✗ CUDA compilation will be DISABLED")
    if not torch.cuda.is_available():
        print("  - torch.cuda.is_available() is False")
    if CUDA_HOME is None:
        print("  - CUDA_HOME is None")

