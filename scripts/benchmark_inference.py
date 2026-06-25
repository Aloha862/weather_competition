"""Run inference with benchmark output enabled.

Example:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --tta hflip
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark batch inference speed.")
    parser.add_argument("--tta", choices=["none", "hflip"], default="none")
    parser.add_argument("--batch-size", default=None)
    args = parser.parse_args()

    env = os.environ.copy()
    env["INFERENCE_BENCHMARK"] = "true"
    env["USE_TTA"] = "true" if args.tta != "none" else "false"
    env["TTA_MODE"] = args.tta
    if args.batch_size is not None:
        env["BATCH_SIZE"] = str(args.batch_size)
    subprocess.run([sys.executable, "infer.py"], env=env, check=True)


if __name__ == "__main__":
    main()

