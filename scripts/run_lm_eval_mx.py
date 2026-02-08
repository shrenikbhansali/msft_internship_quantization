#!/usr/bin/env python3
import argparse
import json
import os
import sys
import subprocess

from mx import add_mx_args, get_mx_specs, finalize_mx_specs
from mx.specs import get_default_mx_specs

REQUIRED_W = "fp4_e2m1"
REQUIRED_A = "fp6_e2m3"


def usage():
    print(
        """Usage:
  run_lm_eval_mx.py [mx_args...] [--no-custom-cuda] -- <command...>

Examples:
  run_lm_eval_mx.py --block_size 32 --scale_bits 8 --custom_cuda -- \
    python -m lm_eval run --model hf --model_args pretrained=... --tasks lambada_openai

Notes:
- MX args are parsed with microxcaling's add_mx_args/get_mx_specs (README guidance).
- Required formats for Exercise 1/2 are enforced:
  w_elem_format=fp4_e2m1, a_elem_format=fp6_e2m3.
- Use --no-custom-cuda to disable custom CUDA kernels if needed.
"""
    )


def _split_args(argv):
    if "--" not in argv:
        return None, None
    idx = argv.index("--")
    return argv[:idx], argv[idx + 1 :]


def _flag_present(raw_args, name):
    prefix = f"--{name}"
    return any(a == prefix or a.startswith(prefix + "=") for a in raw_args)


def build_mx_specs(mx_args, disable_custom_cuda):
    parser = argparse.ArgumentParser(add_help=False)
    parser = add_mx_args(parser)
    args = parser.parse_args(mx_args)

    defaults = get_default_mx_specs()

    # Baseline specs aligned to Exercise 1/2 requirements
    specs = {
        "scale_bits": 8,
        "block_size": 32,
        "bfloat": 16,
        "custom_cuda": not disable_custom_cuda,
        "w_elem_format": REQUIRED_W,
        "a_elem_format": REQUIRED_A,
    }

    # Apply only explicitly provided overrides
    for key, default_val in defaults.items():
        if isinstance(default_val, bool) and default_val is True:
            if _flag_present(mx_args, f"no_{key}"):
                specs[key] = False
        elif isinstance(default_val, bool) and default_val is False:
            if _flag_present(mx_args, key):
                specs[key] = True
        else:
            if _flag_present(mx_args, key):
                specs[key] = getattr(args, key)

    # Enforce required formats
    if _flag_present(mx_args, "w_elem_format") and specs.get("w_elem_format") != REQUIRED_W:
        raise SystemExit(f"ERROR: w_elem_format must be {REQUIRED_W} for Exercise 1/2.")
    if _flag_present(mx_args, "a_elem_format") and specs.get("a_elem_format") != REQUIRED_A:
        raise SystemExit(f"ERROR: a_elem_format must be {REQUIRED_A} for Exercise 1/2.")

    specs["w_elem_format"] = REQUIRED_W
    specs["a_elem_format"] = REQUIRED_A

    specs = finalize_mx_specs(specs, early_exit=False)
    return specs


def main(argv):
    mx_args, cmd = _split_args(argv)
    if mx_args is None or not cmd:
        usage()
        return 1

    disable_custom_cuda = False
    if "--no-custom-cuda" in mx_args:
        disable_custom_cuda = True
        mx_args = [a for a in mx_args if a != "--no-custom-cuda"]

    specs = build_mx_specs(mx_args, disable_custom_cuda)
    env = os.environ.copy()
    env["MX_SPECS_JSON"] = json.dumps(dict(specs))

    # Exec command
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
