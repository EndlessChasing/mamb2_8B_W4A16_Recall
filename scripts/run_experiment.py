#!/usr/bin/env python3
"""Run the declared quantization, discarded smoke, final training and paired evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--prose-tokens',type=Path,required=True)
    args=p.parse_args()
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    receipt=logs/'experiment_v1.json'
    if receipt.exists(): raise FileExistsError(receipt)
    base=ROOT/'artifacts/w4_base_v1'
    train=ROOT/'artifacts/w4_resurface_v1'
    data=ROOT/'training_data/numeric_v1'
    report=ROOT/'reports/w4_resurface_v1_confirm_full.json'
    report.parent.mkdir(exist_ok=True)
    common=['--source-dir',str(args.source_dir),'--w4-dir',str(base),'--data-root',str(data),
            '--train-manifest-sha256',sha(data/'train/manifest.json'),
            '--prose-manifest',str(ROOT/'docs/prose_train_manifest.json'),
            '--prose-tokens',str(args.prose_tokens)]
    stages=[
      ('quantize',['scripts/quantize_w4.py','--source',str(args.source_dir),'--output',str(base)]),
      ('smoke',['scripts/train.py',*common,'--out-dir',str(ROOT/'artifacts/w4_smoke_v1'),'--smoke']),
      ('train',['scripts/train.py',*common,'--out-dir',str(train)]),
      ('evaluate',['scripts/evaluate.py','--source-dir',str(args.source_dir),'--w4-dir',str(base),
                   '--adapter',str(train/'adapter_fp16.pt'),'--train-report',str(train/'report.json'),
                   '--data-root',str(data),'--eval-manifest-sha256',sha(data/'confirm/manifest.json'),
                   '--report',str(report)]),
    ]
    state={'started_unix':time.time(),'complete':False,'stages':[],
           'protocol_sha256':sha(ROOT/'docs/PROTOCOL.md'),
           'code_sha256':{str(f.relative_to(ROOT)):sha(f) for folder in ['mamba2_recall','scripts']
                          for f in sorted((ROOT/folder).glob('*.py'))}}
    def save():
        tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(receipt)
    save()
    env=dict(os.environ, PYTHONPATH=str(ROOT))
    for name,cmd in stages:
        row={'name':name,'command':[sys.executable,*cmd],'started_unix':time.time()}
        state['stages'].append(row);state['current_stage']=name;save()
        log=logs/(name+'_v1.log')
        with log.open('xb') as f:
            result=subprocess.run(row['command'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                  stdout=f,stderr=subprocess.STDOUT)
        row.update(returncode=result.returncode,finished_unix=time.time(),log=str(log))
        save()
        if result.returncode:
            state['failed_stage']=name;save();raise SystemExit(result.returncode)
    state.update(complete=True,finished_unix=time.time());save()

if __name__=='__main__':main()
