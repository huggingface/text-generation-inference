import subprocess

def is_cuda_system():
    try:
        subprocess.check_output("nvidia-smi")
        return True
    except Exception:
        return False

def is_rocm_system():
    try:
        subprocess.check_output("rocm-smi")
        return True
    except Exception:
        return False