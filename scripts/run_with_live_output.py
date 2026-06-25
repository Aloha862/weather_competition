"""Run a command and stream child-process output line by line.

Useful in Jupyter when plain subprocess.run does not show progress promptly.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a command with live output.")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        raise SystemExit("Usage: python scripts/run_with_live_output.py -- python train.py")
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()

