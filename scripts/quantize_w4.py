#!/usr/bin/env python3
"""Build the independent, weight-only MSE-fitted W4 package."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mamba2_recall import runtime
from mamba2_recall.w4 import DEFAULT_CHUNK_ROWS, quantize_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Pinned official NVIDIA checkpoint file or directory")
    parser.add_argument("--output", required=True, help="Packed package directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--protocol", default=str(Path(__file__).resolve().parents[1] / "docs" / "PROTOCOL.md"))
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.chunk_rows < 1:
        parser.error("CPU threads and chunk rows must be positive")
    torch.set_num_threads(args.cpu_threads)
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    start = time.monotonic()

    def progress(event):
        print(json.dumps({"event": "tensor_complete", "elapsed_seconds": time.monotonic() - start, **event}), flush=True)

    manifest = quantize_package(args.source, args.output, device=args.device,
                                chunk_rows=args.chunk_rows, resume=args.resume,
                                progress=progress, protocol_path=args.protocol)
    receipt = {"event": "package_complete", "elapsed_seconds": time.monotonic() - start,
               "manifest_sha256": runtime.sha256_file(Path(args.output) / "manifest.json"),
               "tensor_count": manifest["tensor_count"], "tensor_bytes": manifest["tensor_bytes"],
               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith("cuda") else None,
               "complete": manifest["complete"]}
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
