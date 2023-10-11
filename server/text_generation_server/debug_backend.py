# 	SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=2 text_generation_server/cli.py serve distilgpt2
import subprocess
import os
from pathlib import Path

wd_dir: str = Path(__file__).parent.absolute()
cli_path: str = os.path.join(wd_dir, "cli.py")
os.environ["SAFETENSORS_FAST_GPU"] = "1"
command: str = f"python -m torch.distributed.run --nproc_per_node=1 {cli_path} serve bigscience/bloom-560m"
subprocess.run(command.split())