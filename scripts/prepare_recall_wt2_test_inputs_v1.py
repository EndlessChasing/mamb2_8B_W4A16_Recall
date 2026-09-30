#!/usr/bin/env python3
"""CPU helpers: assemble released runtime files or recreate WT2 test tokens."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
from run_recall_native_prefill_wt2_test_v1 import PINS, PINS_SHA256, need, put, sha


def fixed_pins():
    need(sha(PINS) == PINS_SHA256, 'Frozen test pins changed')
    return json.loads(PINS.read_text())


def prepare_runtime(args):
    fixed = fixed_pins()['models'][args.model_kind]
    need(not args.out_dir.exists(), 'Fresh runtime output required')
    source = args.download_root.resolve()
    planned = {}
    for relative, expected in fixed['frozen_files'].items():
        # The original W4 Hub bundle has code/ but keeps evidence in reports/.
        code_candidate = source / 'code' / relative
        path = code_candidate if code_candidate.is_file() else source / relative
        need(path.is_file() and sha(path) == expected, 'Published runtime/evidence differs: ' + relative)
        planned[relative] = path
    adapter = source / fixed['adapter_path']
    if not adapter.is_file() and args.model_kind == 'w4':
        adapter = source / 'adapter/adapter_fp16.pt'
    need(adapter.is_file() and sha(adapter) == fixed['adapter']['sha256'], 'Published adapter differs')
    planned[fixed['adapter_path']] = adapter
    args.out_dir.mkdir(parents=True)
    for relative, path in planned.items():
        target = args.out_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    receipt = {'complete': True, 'model_kind': args.model_kind, 'pins_sha256': sha(PINS),
               'source': str(source), 'runtime_root': str(args.out_dir.resolve()),
               'files': {relative: {'bytes': path.stat().st_size, 'sha256': sha(path)}
                         for relative, path in planned.items()},
               'weight_files_copied': False, 'cuda_initialized': False, 'torch_imported': 'torch' in sys.modules}
    put(args.out_dir / 'test_runtime_receipt.json', receipt)
    print(json.dumps(receipt, indent=2))


def recreate_tokens(args):
    pins = fixed_pins()
    release = args.release_root.resolve()
    fixed = pins['models'][args.model_kind]
    for relative in ('mamba2_recall/__init__.py', 'mamba2_recall/runtime.py', 'mamba2_recall/calibration.py'):
        need(sha(release / relative) == fixed['frozen_files'][relative], 'Frozen tokenizer/data source changed')
    need('mamba2_recall' not in sys.modules, 'A different runtime was imported')
    sys.path.insert(0, str(release))
    from mamba2_recall import runtime, calibration
    import torch
    need(not torch.cuda.is_initialized(), 'Token reconstruction must be CPU only')
    tokenizer = runtime.SentencePieceTokenizer(args.tokenizer_source_dir)
    ids, metadata = calibration.load_wikitext_tokens(tokenizer, 'test')
    need(metadata == pins['test_dataset'], 'Pinned official test tokenization differs')
    raw = ids.numpy().astype('<i8', copy=False).tobytes()
    need(hashlib.sha256(raw).hexdigest() == metadata['token_stream_sha256_int64le'], 'Raw token identity differs')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / 'tokens.int64le'
    if path.exists():
        need(sha(path) == metadata['token_stream_sha256_int64le'], 'Existing raw token file differs')
    else:
        path.write_bytes(raw)
    need(not torch.cuda.is_initialized(), 'CPU helper initialized CUDA')
    receipt = {'complete': True, 'model_kind': args.model_kind, 'dataset': metadata,
               'file': str(path.resolve()), 'bytes': len(raw), 'sha256': sha(path),
               'cuda_initialized': False, 'weight_checkpoint_loaded': False}
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    runtime = sub.add_parser('runtime', help='Small frozen files from the original published bundle; no weights copied')
    runtime.add_argument('--model-kind', choices=('fp16', 'w4'), required=True)
    runtime.add_argument('--download-root', type=Path, required=True)
    runtime.add_argument('--out-dir', type=Path, required=True)
    tokens = sub.add_parser('tokens', help='Recreate unpublished raw tokens for the CPU evidence audit')
    tokens.add_argument('--model-kind', choices=('fp16', 'w4'), required=True)
    tokens.add_argument('--release-root', type=Path, required=True)
    tokens.add_argument('--tokenizer-source-dir', type=Path, required=True)
    tokens.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    (prepare_runtime if args.command == 'runtime' else recreate_tokens)(args)


if __name__ == '__main__':
    main()
