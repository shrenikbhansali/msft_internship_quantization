#!/usr/bin/env python3
import os
import sys
import time
import subprocess
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WRAPPER = os.path.join(SCRIPT_DIR, "run_lm_eval_mx.py")

RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "/content/results/ex2")
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "32")
DEVICE = os.environ.get("DEVICE", "cuda:0")

BLOCK_SIZES = [16, 32, 64]
SCALE_BITS = [8, 6]
CUSTOM_CUDA_FLAGS = [True, False]


def write_env(run_dir):
    env_path = os.path.join(run_dir, "env.txt")
    with open(env_path, "w") as f:
        f.write(f"date: {datetime.now().isoformat()}\n")
        f.flush()
        subprocess.run(
            [sys.executable, "-"],
            input=(
                "import torch, transformers\n"
                "print('torch:', torch.__version__)\n"
                "print('transformers:', transformers.__version__)\n"
                "print('cuda available:', torch.cuda.is_available())\n"
            ),
            text=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
        subprocess.call(["nvidia-smi", "-L"], stdout=f, stderr=subprocess.STDOUT)
        subprocess.call(["git", "-C", "/content/transformers", "rev-parse", "HEAD"], stdout=f, stderr=subprocess.STDOUT)


def run_one(bs, sb, cc):
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_bs{bs}_sb{sb}_cc{str(cc).lower()}"
    run_dir = os.path.join(RESULTS_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)

    cuda_flag = "--custom_cuda" if cc else "--no-custom-cuda"

    cmd = [
        sys.executable,
        WRAPPER,
        cuda_flag,
        "--block_size",
        str(bs),
        "--scale_bits",
        str(sb),
        "--",
        "python",
        "-m",
        "lm_eval",
        "run",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={MODEL_ID}",
        "--tasks",
        "lambada_openai",
        "--limit",
        "0.25",
        "--device",
        DEVICE,
        "--batch_size",
        str(BATCH_SIZE),
        "--output_path",
        os.path.join(run_dir, "metrics.json"),
    ]

    with open(os.path.join(run_dir, "command.txt"), "w") as f:
        f.write(" ".join(cmd) + "\n")

    write_env(run_dir)

    output_path = os.path.join(run_dir, "output.log")
    start = time.time()
    with open(output_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        ret = proc.wait()
        elapsed = time.time() - start
        f.write(f"\n[run_ex2_ablations] wall_time_seconds={elapsed:.3f}\n")
    return ret


def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    for bs in BLOCK_SIZES:
        for sb in SCALE_BITS:
            for cc in CUSTOM_CUDA_FLAGS:
                rc = run_one(bs, sb, cc)
                if rc != 0:
                    print(f"Run failed (bs={bs}, sb={sb}, cc={cc}).", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
