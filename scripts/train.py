#!/usr/bin/env python3
"""Train Resurface on a frozen independent W4A16 Mamba2-8B base."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba2_recall import resurface_data as data, resurface_native as native
from mamba2_recall import resurface_loss as objective
from mamba2_recall import runtime
from mamba2_recall.w4 import load_w4_model

STEPS = 1536
PROSE_MANIFEST_SHA = 'facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d'
PROSE_TOKENS_SHA = 'e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def lr_factor(j):
    return .1 + .9*(1+math.cos(math.pi*j/(STEPS-1)))/2


def schedule():
    return torch.randperm(STEPS, generator=torch.Generator(device='cpu').manual_seed(2026092803)).tolist()


def load_training_inputs(args):
    tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
    manifest, _, examples = data.load_training(args.data_root,
        args.train_manifest_sha256, tokenizer)
    protocol_sha = data.sha_file(ROOT/'docs'/'PROTOCOL.md')
    if manifest['protocol_sha256'] != protocol_sha or len(examples) != STEPS:
        raise ValueError('Prepared numeric TRAIN split differs from frozen protocol')
    if data.sha_file(args.prose_manifest) != PROSE_MANIFEST_SHA:
        raise ValueError('Pinned WikiText TRAIN manifest differs')
    prose_manifest = json.loads(args.prose_manifest.read_text())
    windows = torch.load(args.prose_tokens, map_location='cpu', weights_only=True)
    prose_file_sha = data.sha_file(args.prose_tokens)
    if (not prose_manifest.get('complete') or
            prose_manifest.get('training_tokens_file_sha256') != PROSE_TOKENS_SHA or
            tuple(windows.shape) != (448,2048) or windows.dtype != torch.int64 or
            sorted(prose_manifest['schedule']) != list(range(448)) or
            runtime.token_digest(windows.flatten().numpy()) !=
                prose_manifest['training_tokens_sha256_int64le']):
        raise ValueError('Expected 448 disjoint, pinned prose TRAIN windows')
    w4_manifest = json.loads((args.w4_dir/'manifest.json').read_text())
    if (not w4_manifest['complete']
            or w4_manifest['source_checkpoint_sha256'] != runtime.SOURCE_CHECKPOINT_SHA256
            or w4_manifest.get('protocol_sha256') != protocol_sha):
        raise ValueError('W4 package source or completeness differs')
    binding = {'w4_manifest_sha256':data.sha_file(args.w4_dir/'manifest.json'),
               'source_checkpoint_sha256': runtime.SOURCE_CHECKPOINT_SHA256,
               'tokenizer_sha256': tokenizer.sha256,
               'protocol_sha256': protocol_sha,
               'train_manifest_sha256': args.train_manifest_sha256,
               'prose_manifest_sha256': PROSE_MANIFEST_SHA,
               'prose_tokens_sha256': prose_file_sha,
               'prose_tokens_int64le_sha256': prose_manifest['training_tokens_sha256_int64le'],
               'prose_file_exact_historical': prose_file_sha == PROSE_TOKENS_SHA,
               'adapter': native.FORMAT, 'successful_updates': STEPS}
    return tokenizer, examples, windows, prose_manifest['schedule'], binding


def pair_for(j, ordered, examples, windows, prose_order):
    row = examples[ordered[j]]
    full = torch.tensor(row['full_ids'], dtype=torch.long, device='cuda')[None]
    answer_mask = torch.zeros_like(full[:,:-1], dtype=torch.bool)
    answer_mask[:,row['ce_hidden_positions']] = True
    if int(answer_mask.sum()) != row['answer_target_count']:
        raise ValueError('Numeric answer suffix mask differs')
    prose_index = prose_order[j%448]
    prose_start = 512*((j//448)%4)
    prose = windows[prose_index,prose_start:prose_start+512].to('cuda')[None]
    if tuple(prose.shape) != (1,512):
        raise ValueError('Prose segment geometry differs')
    return {'id': row['id'], 'schedule_entry': ordered[j],
            'prose_window': prose_index, 'prose_start': prose_start,
            'mk_ids': full[:,:-1], 'mk_targets': full[:,1:], 'answer_mask': answer_mask,
            'prose_ids': prose[:,:-1], 'prose_targets': prose[:,1:]}


def optimizer_for(bank):
    mix = [p for name,p in bank.masters.items() if name.endswith(('.V_read','.g_read'))]
    router = [p for name,p in bank.masters.items() if name.endswith(('.router_w','.router_b'))]
    return torch.optim.AdamW([{'params':mix,'lr':1e-4,'base_lr':1e-4},
                              {'params':router,'lr':3e-4,'base_lr':3e-4}],
                             betas=(.9,.999),eps=1e-8,weight_decay=0.)


def attempt(bank, teacher, pair, optimizer, scaler, j):
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = group['base_lr']*lr_factor(j)
    hidden,gates = bank.forward_hidden(pair['mk_ids'], use_checkpoint=True)
    mk = objective.staged_loss_backward(hidden,pair['mk_targets'],bank.model.lm_head,
        gates=gates,scaler=scaler,answer_mask=pair['answer_mask'],ce_weight=1.,
        kl_weight=0.,closure_weight=0.,chunk_tokens=64)
    del hidden,gates
    with torch.no_grad():
        teacher_hidden = teacher.backbone(pair['prose_ids'])
    hidden,gates = bank.forward_hidden(pair['prose_ids'],use_checkpoint=True)
    prose = objective.staged_loss_backward(hidden,pair['prose_targets'],bank.model.lm_head,
        gates=gates,scaler=scaler,teacher_hidden=teacher_hidden,
        teacher_head=teacher.lm_head,ce_weight=.5,kl_weight=.5,
        closure_weight=3.,chunk_tokens=64)
    del hidden,gates,teacher_hidden
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in bank.parameters()]
    if any(g is None for g in gradients):
        raise RuntimeError('Missing adapter gradient')
    overflow = (not all(bool(torch.isfinite(g).all()) for g in gradients)
                or not mk['scaled_hidden_gradient_finite']
                or not prose['scaled_hidden_gradient_finite'])
    if overflow:
        scaler.update(new_scale=scaler.get_scale()*.5)
        magnitude = None
    else:
        magnitude = float(torch.nn.utils.clip_grad_norm_(bank.parameters(),1.,error_if_nonfinite=True))
        scaler.step(optimizer)
        scaler.update()
        if any(not bool(torch.isfinite(p).all()) or not bool(torch.isfinite(p.half()).all())
               for p in bank.parameters()):
            raise FloatingPointError('Nonfinite updated adapter master or FP16 cast')
    bank.assert_base_frozen()
    return {'mk_ce':mk['ce'], 'prose_ce':prose['ce'],
            'prose_kl':prose['teacher_to_student_kl'],
            'prose_closure':prose['closure'], 'overflow':overflow,
            'gradient_norm_before_clip':magnitude, 'loss_scale':scaler.get_scale()}


def save_checkpoint(path, bank, optimizer, scaler, binding, success, attempts):
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix('.pt.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open('wb') as stream:
        torch.save({'format':'MAMBA2_W4_RESURFACE_CHECKPOINT_V1',
                    'binding':binding,'successful_updates':success,'attempts':attempts,
                    'masters':bank.state_dict(),'optimizer':optimizer.state_dict(),
                    'scaler':scaler.state_dict()}, stream)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':data.sha_file(path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--w4-dir',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--train-manifest-sha256',required=True)
    p.add_argument('--prose-manifest',type=Path,required=True)
    p.add_argument('--prose-tokens',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--smoke',action='store_true',help='Discard one GPU update; no adapter export')
    args = p.parse_args()
    if args.out_dir.exists():
        raise FileExistsError('New output directory required')
    args.out_dir.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.manual_seed(2026092803); torch.cuda.manual_seed_all(2026092803)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    report = {'format':'MAMBA2_W4_RESURFACE_TRAIN_V1','complete':False,
              'mode':'smoke' if args.smoke else 'formal',
              'started_unix':time.time(),'source_sha256':data.sha_file(__file__),
              'history':[],'checkpoints':[]}
    try:
        tokenizer, examples, windows, prose_order, binding = load_training_inputs(args)
        report['binding'] = binding
        write_json(args.out_dir/'report.json',report)
        student = load_w4_model(args.w4_dir)
        teacher = load_w4_model(args.w4_dir)
        if sum(p.numel() for p in student.parameters()) != 8236999680:
            raise RuntimeError('Wrong W4 decoded model parameter count')
        bank = native.ResurfaceNative(student,'soft')
        if len(bank.masters)!=224 or sum(p.numel() for p in bank.parameters())!=1154104:
            raise RuntimeError('Wrong adapter geometry')
        report['w4_package'] = student._package_receipt
        probe_ids = torch.tensor(tokenizer.encode('The purpose of this check is to verify the frozen model.'),device='cuda')[None]
        with torch.no_grad():
            expected = teacher.backbone(probe_ids)
            actual, _ = bank.forward_hidden(probe_ids,use_checkpoint=False)
        if not torch.equal(expected,actual):
            raise RuntimeError('Zero adapter differs from the independently loaded teacher')
        report['zero_adapter_hidden_exact'] = True
        optimizer = optimizer_for(bank)
        scaler = torch.amp.GradScaler('cuda',init_scale=1024.,growth_factor=2.,
                                      backoff_factor=.5,growth_interval=2000)
        ordered = schedule()
        successful = attempts = 0
        limit = 1 if args.smoke else STEPS
        while successful < limit:
            if attempts >= limit+8:
                raise RuntimeError('Overflow retry budget exceeded')
            pair = pair_for(successful,ordered,examples,windows,prose_order)
            start = time.monotonic()
            result = attempt(bank,teacher,pair,optimizer,scaler,successful)
            attempts += 1
            if not result['overflow']:
                successful += 1
            row = {'attempt':attempts,'successful_updates':successful,
                   'case_id':pair['id'],'schedule_entry':pair['schedule_entry'],
                   'prose_window':pair['prose_window'],'prose_start':pair['prose_start'],
                   'answer_targets':int(pair['answer_mask'].sum()),
                   'seconds':time.monotonic()-start,**result}
            report['history'].append(row)
            if not args.smoke and successful and successful%384==0 and not result['overflow']:
                report['checkpoints'].append(save_checkpoint(args.out_dir/f'checkpoint_{successful:04d}.pt',
                    bank,optimizer,scaler,binding,successful,attempts))
            if attempts%16==0 or successful==limit or result['overflow']:
                report.update(successful_updates=successful,attempts=attempts,
                              gpu_memory=runtime.gpu_memory_receipt())
                write_json(args.out_dir/'report.json',report)
                print(json.dumps({'updates':successful,'attempts':attempts,
                                  'mk_ce':result['mk_ce'],'prose_ce':result['prose_ce'],
                                  'overflow':result['overflow']}),flush=True)
        if not args.smoke:
            adapter_path = args.out_dir/'adapter_fp16.pt'
            exported = bank.export_fp16(adapter_path,binding=binding)
            if native.read_fp16(adapter_path,expected_binding=binding)['gate_mode']!='soft':
                raise RuntimeError('Export roundtrip differs')
            report['adapter'] = exported
            report['frozen_base_check'] = bank.assert_base_frozen()
            report['teacher_base_parameters_frozen'] = all(
                not p.requires_grad and p.grad is None for p in teacher.parameters())
        report.update(complete=True,successful_updates=successful,attempts=attempts,
                      gpu_memory=runtime.gpu_memory_receipt())
        bank.close()
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report['finished_unix'] = time.time()
        write_json(args.out_dir/'report.json',report)


if __name__=='__main__':
    main()
