#!/usr/bin/env python3
"""Check the codec or every file and decoded hash in a completed W4 package."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mamba2_recall import runtime, w4


def _raises(call, exceptions=(ValueError, TypeError)):
    try:
        call()
    except exceptions:
        return
    raise AssertionError("Expected invalid input to be rejected")


def self_test():
    """Exercise independent bit patterns, edges, cross-row chunks and corruption."""
    torch.manual_seed(20260928)
    codes = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
    packed = w4.pack_nibbles(codes)
    expected = torch.tensor([[0x10, 0x32, 0x54, 0x76], [0x98, 0xba, 0xdc, 0xfe]], dtype=torch.uint8)
    assert torch.equal(packed, expected)
    assert torch.equal(w4.unpack_nibbles(expected, 8), codes)
    odd = torch.tensor([[0, 15, 3], [9, 2, 14]], dtype=torch.uint8)
    assert torch.equal(w4.unpack_nibbles(w4.pack_nibbles(odd), 3), odd)
    _raises(lambda: w4.pack_nibbles(torch.tensor([[16]], dtype=torch.uint8)))
    _raises(lambda: w4.unpack_nibbles(torch.tensor([[0xff]], dtype=torch.uint8), 1))
    for fill in (0., 1.25, -7.5, torch.finfo(torch.float16).smallest_normal / 1024):
        weights = torch.full((3, 257), fill, dtype=torch.float16)
        scales, offsets, codes, stats = w4.quantize_groups(weights)
        assert torch.equal(w4.decode_groups(scales, offsets, codes), weights)
        assert stats["squared_error"] == 0
        assert bool((scales == 0).all())
    for value in (float("nan"), float("inf"), -float("inf"), 100000.):
        _raises(lambda: w4.quantize_groups(torch.tensor([[value]], dtype=torch.float32)))
    weights = (torch.randn(7, 259) * .07).half()
    weights[0, :4] = torch.tensor([0., 0.000000059604645, -0.000000059604645, 0.])
    scales, offsets, codes, stats = w4.quantize_groups(weights)
    decoded = w4.decode_groups(scales, offsets, codes)
    actual_sse = (decoded.float() - weights.float()).square().double().sum().item()
    assert abs(actual_sse - stats["squared_error"]) <= 1e-6 * max(1., actual_sse)
    assert stats["squared_error"] <= stats["unclipped_squared_error"] + 1e-8
    assert sum(stats["candidate_group_counts"]) == stats["group_count"]
    # Saturated FP16 endpoints must yield a finite selected representation.
    extreme = torch.tensor([[-65504., 65504.] * 64], dtype=torch.float16)
    sx, ox, cx, _ = w4.quantize_groups(extreme)
    assert bool(torch.isfinite(w4.decode_groups(sx, ox, cx)).all())
    with tempfile.TemporaryDirectory(prefix="mamba2-w4-check-") as temporary:
        directory = Path(temporary)
        path = directory / "0000.w4bin"
        entry = w4.write_tensor(path, weights, "w4_affine_f16", chunk_rows=2, reserve_bytes=0)
        restored = w4.read_tensor(path, chunk_rows=3)
        assert torch.equal(restored, decoded)
        assert entry["decoded_sha256"] == hashlib.sha256(w4._bytes(decoded)).hexdigest()
        w4.verify_tensor_entry(directory, entry)
        fp16_path = directory / "0001.w4bin"
        fp16_source = torch.randn(3, 4, 5).bfloat16()
        fp16_entry = w4.write_tensor(fp16_path, fp16_source, "fp16", reserve_bytes=0)
        assert torch.equal(w4.read_tensor(fp16_path, chunk_rows=2), fp16_source.half())
        w4.verify_tensor_entry(directory, fp16_entry)
        data = path.read_bytes()
        for length in (0, w4.PREFIX.size - 1, w4.PREFIX.size + 1, len(data) - 1):
            bad = directory / "truncated.w4bin"
            bad.write_bytes(data[:length])
            _raises(lambda: w4.read_tensor(bad))
        bad.write_bytes(data + b"\x00")
        _raises(lambda: w4.read_tensor(bad))
        corrupt = bytearray(data)
        corrupt[-2] ^= 1
        path.write_bytes(corrupt)
        _raises(lambda: w4.verify_tensor_entry(directory, entry))
        path.write_bytes(data)
        # Manifest checks fail before any model allocation for wrong completion,
        # source, config, algorithm, counts or coverage.
        base = {"format": w4.FORMAT, "version": 1, "complete": True,
                "source_checkpoint_sha256": runtime.SOURCE_CHECKPOINT_SHA256,
                "model_config": copy.deepcopy(runtime.MODEL_CONFIG),
                "quantization": copy.deepcopy(w4.QUANTIZATION),
                "parameter_count": w4.PARAMETER_COUNT, "tensor_count": w4.TENSOR_COUNT,
                "w4_tensor_count": w4.W4_TENSOR_COUNT, "fp16_tensor_count": 393,
                "tensors": {}, "tensor_bytes": 0}
        for key, value in (("complete", False), ("source_checkpoint_sha256", "0" * 64),
                           ("model_config", {}), ("quantization", {}), ("tensor_count", 506),
                           ("parameter_count", w4.PARAMETER_COUNT - 1)):
            invalid = copy.deepcopy(base)
            invalid[key] = value
            _raises(lambda: w4.validate_manifest(invalid, expected_shapes={}))
        _raises(lambda: w4.validate_manifest(base, expected_shapes={}))
    return {"passed": True, "checks": ["all_nibble_codes_and_byte_order", "odd_columns_and_padding",
            "zero_constant_subnormal_groups", "nonfinite_rejection", "fp16_endpoint_overflow",
            "serialized_mse_selection", "cross_chunk_row_roundtrip", "fp16_roundtrip",
            "truncation_and_trailing_bytes", "file_and_decoded_hashes", "strict_invalid_manifest"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--package")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--load-model", action="store_true", help="Also instantiate and strictly load the FP16 native model")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.self_test and not args.package:
        parser.error("Choose --self-test and/or --package")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    if args.self_test:
        print(json.dumps({"event": "codec_self_test", **self_test()}), flush=True)
    if args.package:
        manifest = w4.read_manifest(args.package)
        for index, (name, entry) in enumerate(manifest["tensors"].items(), 1):
            w4.verify_tensor_entry(args.package, entry)
            print(json.dumps({"event": "tensor_verified", "index": index, "tensor": name}), flush=True)
        result = {"event": "package_verified", "tensor_count": len(manifest["tensors"]),
                  "manifest_sha256": runtime.sha256_file(Path(args.package) / "manifest.json"),
                  "tensor_bytes": manifest["tensor_bytes"]}
        if args.load_model:
            model = w4.load_w4_model(args.package, device=args.device)
            result["model_receipt"] = model._package_receipt
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
