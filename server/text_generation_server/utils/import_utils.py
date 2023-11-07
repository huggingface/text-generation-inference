import subprocess

IS_CUDA_SYSTEM = False
IS_ROCM_SYSTEM = False

try:
    subprocess.check_output("nvidia-smi")
    IS_CUDA_SYSTEM = True
except Exception:
    IS_CUDA_SYSTEM = False

try:
    subprocess.check_output("rocm-smi")
    IS_ROCM_SYSTEM = True
except Exception:
    IS_ROCM_SYSTEM = False