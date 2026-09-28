#!/usr/bin/env python3
"""Paired independent W4A16 base/readout recall and WikiText-2 validation evaluation."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mamba2_recall import calibration, evaluation, resurface_data as data
from mamba2_recall import resurface_native as native, runtime
from mamba2_recall.w4 import load_w4_model

MATCH = re.compile(r'(?<!\d)\d{6}(?!\d)')


def write_json(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


@torch.inference_mode()
def score_mk(model,tokenizer,cases,limit=None):
    rows=[]
    for index,case in enumerate(cases[:limit] if limit is not None else cases):
        output,generated,tokens,cache_bytes = evaluation.generate_greedy(
            model,tokenizer,case['prompt'],max_new_tokens=12,execution='prefill')
        match = MATCH.search(output)
        prediction = match.group() if match else None
        rows.append({'id':case['id'],'condition':case['condition'],
            'N':case['N'],'template':case['template'],
            'prompt_token_sha256_int64le':data.token_digest(tokenizer.encode(case['prompt'])),
            'expected':case['answer'],'prediction':prediction,
            'correct':prediction==case['answer'],'generated_ids':generated,
            'output':output,'prompt_tokens':tokens,'cache_bytes':cache_bytes})
        if cache_bytes!=122028032:
            raise RuntimeError('FP16 native cache bytes differ')
        if (index+1)%32==0 or index==0:
            print(json.dumps({'mk_completed':index+1,'mk_total':len(cases) if limit is None else limit}),flush=True)
    summary={}
    for condition in ('normal','target_removed'):
        selected=[r for r in rows if r['condition']==condition]
        if selected:
            summary[condition]={'correct':sum(r['correct'] for r in selected),
                'count':len(selected),'accuracy':sum(r['correct'] for r in selected)/len(selected)}
    return {'rows':rows,'summary':summary,'generation':'greedy max12, full256K, fresh FP16 cache',
            'matching':'first standalone six-digit integer'}


def compare_mk(baseline,active):
    result={}
    for condition in ('normal','target_removed'):
        counts={'both_correct':0,'both_wrong':0,'gained':0,'lost':0}
        a=[row for row in baseline['rows'] if row['condition']==condition]
        b=[row for row in active['rows'] if row['condition']==condition]
        if len(a)!=len(b):
            raise ValueError('MK arm length differs')
        for before,after in zip(a,b):
            if any(before[key]!=after[key] for key in ('id','condition','expected','prompt_token_sha256_int64le')):
                raise ValueError('Paired MK prompt identity differs')
            bucket={(False,False):'both_wrong',(True,True):'both_correct',
                    (False,True):'gained',(True,False):'lost'}[(before['correct'],after['correct'])]
            counts[bucket]+=1
        result[condition]={**counts,'count':len(a),
            'baseline_correct':sum(r['correct'] for r in a),
            'adapter_correct':sum(r['correct'] for r in b),
            'delta_percentage_points':100*(counts['gained']-counts['lost'])/len(a)}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--w4-dir',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--eval-manifest-sha256',required=True)
    p.add_argument('--split',choices=('dev','confirm'),default='confirm')
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--smoke',action='store_true',help='First 8 MK rows and 2 prose windows only')
    args=p.parse_args()
    if args.report.exists():
        raise FileExistsError('Preserve existing report')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
    manifest,cases,tokens=data.load_evaluation(args.data_root,args.split,args.eval_manifest_sha256,tokenizer)
    if (manifest['protocol_sha256']!=data.sha_file(ROOT/'docs'/'PROTOCOL.md')
            or any(data.token_digest(tokenizer.encode(row['prompt']))!=tok['prompt_token_sha256_int64le']
                   for row,tok in zip(cases,tokens))):
        raise ValueError('Evaluation data/protocol identity differs')
    adapter=native.read_fp16(args.adapter)
    w4_manifest_sha256=data.sha_file(args.w4_dir/'manifest.json')
    if adapter['binding'].get('w4_manifest_sha256')!=w4_manifest_sha256:
        raise ValueError('Adapter bound to another W4 package')
    if adapter['binding'].get('source_checkpoint_sha256')!=runtime.SOURCE_CHECKPOINT_SHA256:
        raise ValueError('Adapter bound to another base')
    if adapter['binding'].get('protocol_sha256')!=manifest['protocol_sha256']:
        raise ValueError('Adapter and evaluation split use different protocol')
    report={'format':'MAMBA2_W4_RESURFACE_EVAL_V1','complete':False,
        'started_unix':time.time(),'smoke':args.smoke,'split':args.split,
        'source_checkpoint_sha256':runtime.SOURCE_CHECKPOINT_SHA256,
        'w4_manifest_sha256':w4_manifest_sha256,
        'tokenizer_sha256':tokenizer.sha256,'protocol_sha256':manifest['protocol_sha256'],
        'data_manifest_sha256':args.eval_manifest_sha256,
        'adapter_sha256':data.sha_file(args.adapter),
        'dataset_status':'previously observed template family and numeric CONFIRM instances',
        'model_precision':'independent packed affine W4, decoded FP16 reference; A16 and FP16 cache'}
    write_json(args.report,report)
    try:
        ids,dataset=calibration.load_wikitext_tokens(tokenizer,'validation')
        windows=evaluation.ppl_windows(ids,2048)
        if not args.smoke and (len(windows)!=130 or sum(len(w)-1 for _,w in windows)!=264764):
            raise ValueError('Full WikiText validation target/window count differs')
        if args.smoke:
            windows=windows[:2]
        report['dataset']=dataset
        report['ppl_windows']=len(windows)
        report['mk_cases']=len(cases) if not args.smoke else 8
        model=load_w4_model(args.w4_dir)
        report['w4_package']=model._package_receipt
        report['baseline_ppl']=evaluation.evaluate_ppl(model,windows)
        write_json(args.report,report)
        report['baseline_mk']=score_mk(model,tokenizer,cases,8 if args.smoke else None)
        write_json(args.report,report)
        with native.install_fp16(model,args.adapter,expected_binding=adapter['binding']) as bank:
            report['adapter_ppl']=evaluation.evaluate_ppl(model,windows)
            write_json(args.report,report)
            report['adapter_mk']=score_mk(model,tokenizer,cases,8 if args.smoke else None)
            report['frozen_base_check']=bank.assert_base_frozen()
        report['mk_comparison']=compare_mk(report['baseline_mk'],report['adapter_mk'])
        report['ppl_delta_percent']=100*(report['adapter_ppl']['ppl']/report['baseline_ppl']['ppl']-1)
        report['restored_probe']=score_mk(model,tokenizer,cases,8)
        if report['restored_probe']['rows']!=report['baseline_mk']['rows'][:8]:
            raise RuntimeError('Adapter removal failed exact same-process MK replay')
        if args.smoke:
            report['quality_claim']='smoke only; no full MK or PPL conclusion'
        else:
            report['quality_claim']='full specified protocol, reused known template family; no untouched generalization claim'
        report['gpu_memory']=runtime.gpu_memory_receipt()
        report['complete']=True
        print(json.dumps({'complete':True,'baseline_ppl':report['baseline_ppl']['ppl'],
            'adapter_ppl':report['adapter_ppl']['ppl'],
            'mk':{k:v['delta_percentage_points'] for k,v in report['mk_comparison'].items()}}),flush=True)
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report['finished_unix']=time.time()
        write_json(args.report,report)


if __name__=='__main__':
    main()
