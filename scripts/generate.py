#!/usr/bin/env python3
"""Greedy completion using the decoded W4 base and optional final Resurface adapter."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mamba2_recall import evaluation, resurface_native as native, runtime
from mamba2_recall.w4 import load_w4_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--w4-dir',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--adapter',type=Path)
    p.add_argument('--prompt',required=True)
    p.add_argument('--max-new-tokens',type=int,default=64)
    args=p.parse_args()
    if args.max_new_tokens<1: p.error('--max-new-tokens must be positive')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    tokenizer=runtime.SentencePieceTokenizer(args.tokenizer)
    length=len(tokenizer.encode(args.prompt))
    if not length or length+args.max_new_tokens>4096:
        p.error('Provide a nonempty prompt with prompt+completion length at most4096 tokens')
    binding=None
    if args.adapter:
        payload=native.read_fp16(args.adapter)
        binding=payload['binding']
        if (payload['gate_mode']!='soft' or binding.get('successful_updates')!=1536
                or binding.get('w4_manifest_sha256')!=runtime.sha256_file(args.w4_dir/'manifest.json')
                or binding.get('source_checkpoint_sha256')!=runtime.SOURCE_CHECKPOINT_SHA256
                or binding.get('tokenizer_sha256')!=tokenizer.sha256):
            raise ValueError('Final soft adapter does not match this W4 package/tokenizer')
    model=load_w4_model(args.w4_dir)
    context=native.install_fp16(model,args.adapter,expected_binding=binding) if args.adapter else nullcontext()
    with context:
        completion,ids,prompt_tokens,cache_bytes=evaluation.generate_greedy(
            model,tokenizer,args.prompt,max_new_tokens=args.max_new_tokens,execution='prefill')
    print(json.dumps({'completion':completion,'generated_ids':ids,'prompt_tokens':prompt_tokens,
        'cache_bytes':cache_bytes,'adapter_enabled':args.adapter is not None,
        'runtime':'decoded FP16 reference; packed-resident W4 kernel is not implemented'},ensure_ascii=False))

if __name__=='__main__':main()
