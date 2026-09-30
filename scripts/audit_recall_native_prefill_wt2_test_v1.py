#!/usr/bin/env python3
"""Independent CPU arithmetic and token coverage audit of frozen WT2 test runs."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
from run_recall_native_prefill_wt2_test_v1 import (ARMS, FORMAT, PACKAGES, PINS, PINS_SHA256,
                                                PROTOCOL, need, put, sha)
RUNNER = Path(__file__).with_name('run_recall_native_prefill_wt2_test_v1.py')


def close(actual, expected, label):
    need(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), label)


def audit(directory):
    need(sha(PINS) == PINS_SHA256, 'Frozen test pins changed')
    pins = json.loads(PINS.read_text())
    report = json.loads((directory / 'comparison.json').read_text())
    need(report['format'] == FORMAT and report['complete'], 'Incomplete or unsupported paired report')
    kind = report['model_kind']
    fixed = pins['models'][kind]
    need(report['release_binding'] == fixed and report['pins_sha256'] == sha(PINS) and
         report['protocol_sha256'] == sha(PROTOCOL) and report['runner_sha256'] == sha(RUNNER),
         'Runner/protocol/published checkpoint binding differs')
    need(report['dataset'] == pins['test_dataset'], 'Official test dataset identity differs')
    need(report['checkpoint_frozen_before_test'] is True and report['test_used_for_training_or_selection'] is False,
         'Test used to modify/select the checkpoint')
    backend = report['backend']
    need(backend['packages'] == PACKAGES and backend['cuda_runtime'] == '12.8' and
         backend['num_threads'] == 8 and backend['float32_matmul_precision'] == 'highest' and
         backend['tf32_matmul'] is False, 'Recorded native prefill backend differs')
    need(report['windowing'] == {'targets_per_full_window': 2048, 'logits_chunk_tokens': 64,
         'state_reset_each_window': True, 'final_partial_included': True,
         'state_requantized_each_token': False, 'execution': 'prefill',
         'window_count': pins['window_count'], 'target_tokens': pins['target_tokens']}, 'Window protocol differs')
    token_path = directory / 'tokens.int64le'
    raw = token_path.read_bytes()
    token_receipt = report['tokens_file']
    need(token_receipt == {'file': 'tokens.int64le', 'bytes': len(raw), 'sha256': sha(token_path)} and
         token_receipt['sha256'] == pins['test_dataset']['token_stream_sha256_int64le'] and
         len(raw) == pins['test_dataset']['total_tokens'] * 8, 'Raw saved test tokens differ')
    ids = [v[0] for v in struct.iter_unpack('<q', raw)]
    need(all(0 <= token < 256000 for token in ids), 'Token ID outside original vocabulary')
    expected_windows = []
    for start in range(0, len(ids)-1, 2048):
        end = min(start+2049, len(ids))
        expected_windows.append((start, end-start-1, hashlib.sha256(raw[start*8:end*8]).hexdigest()))
    need(len(expected_windows) == 147 and sum(n for _, n, _ in expected_windows) == 300963,
         'Independent full test target coverage differs')
    ledger_path = directory / 'weight_ledger.json'
    need(report['weight_ledger'] == {'file': 'weight_ledger.json', 'sha256': sha(ledger_path)}, 'Weight ledger changed')
    ledger = json.loads(ledger_path.read_text())
    expected_hashes = ledger['tensor_sha256']
    need(len(expected_hashes) == 507 and ledger['provenance'] == report['weight_provenance'], 'Weight provenance differs')
    if kind == 'w4':
        manifest_path = directory / 'w4_manifest.json'
        need(sha(manifest_path) == fixed['w4_manifest_sha256'], 'Copied published W4 manifest changed')
        manifest = json.loads(manifest_path.read_text())
        need(expected_hashes == {n: v['decoded_sha256'] for n, v in manifest['tensors'].items()} and
             manifest['source_checkpoint_sha256'] == fixed['source_checkpoint_sha256'] and
             manifest['parameter_count'] == 8236999680 and manifest['tensor_count'] == 507 and
             manifest['complete'], 'Expected W4 loaded bytes differ')
        need(report['package_receipt']['manifest_sha256'] == fixed['w4_manifest_sha256'] and
             report['package_receipt']['file_hashes_verified'] and report['package_receipt']['decoded_hashes_verified'],
             'Original strict W4 loader did not verify actual files/decoded weights')
    need(report['package_receipt']['source_checkpoint_sha256'] == fixed['source_checkpoint_sha256'] and
         report['package_receipt']['parameter_count'] == 8236999680, 'Pinned source checkpoint differs')
    for key in ('initial_weight_check', 'final_weight_check'):
        check = report[key]
        need(check['complete'] and check['actual_content_checked'] and check['tensors'] == 507 and
             check['actual_tensor_sha256'] == expected_hashes, 'Actual loaded weight contents differ: ' + key)
    adapter_expected = fixed['adapter']['tensor_sha256']
    def check_adapter(receipt):
        need(receipt == {'complete': True, 'tensor_sha256': adapter_expected,
             'payload_bytes': 2308208, 'parameters': 1154104, 'tensor_count': 224}, 'Actual adapter content differs')
    check_adapter(report['published_adapter_content'])
    probe = report['adapter_removal_probe']
    need(probe['complete'] and probe['matches_unadapted_bitwise'] and probe['tokens'] == 128 and
         probe['hidden_sha256'] == report['unadapted_probe_hidden_sha256'], 'Adapter removal changed native hidden output')
    need(report['final_native_restoration']['complete'] and
         report['final_native_restoration']['forward_methods_and_hooks_exact'], 'Native methods/hooks not restored')
    identity = report['final_base_identity']
    need(identity['complete'] and identity['tensors'] == 507 and identity['parameters'] == 8236999680 and
         identity['identity_version_gradients_unchanged'], 'Base identities changed')
    rows = {}
    for arm in ARMS:
        arm_path = directory / (arm + '.json')
        receipt = report['reports'][arm]
        need(receipt['file'] == arm_path.name and receipt['sha256'] == sha(arm_path), 'Arm report changed')
        row = json.loads(arm_path.read_text())
        adapted = arm == 'resurface'
        need(row['format'] == FORMAT and row['complete'] and row['arm'] == arm and row['model_kind'] == kind and
             row['adapter_loaded'] is adapted and row['dataset'] == pins['test_dataset'] and
             row['adapter_sha256'] == (fixed['adapter']['sha256'] if adapted else None), 'Arm checkpoint identity differs')
        need(row['native_restoration']['complete'] and row['native_restoration']['forward_methods_and_hooks_exact'] and
             row['base_identity']['complete'] and row['base_identity']['identity_version_gradients_unchanged'],
             'Arm did not preserve frozen native model')
        if adapted:
            check_adapter(row['loaded_adapter_content'])
            check = row['frozen_base_check']
            need(check['actual_content_checked'] and check['identity_version_gradients_unchanged'] and
                 check['tensors'] == 507 and check['parameters'] == 8236999680, 'Adapted base not checked')
        ppl = row['ppl']
        need(ppl['execution'] == 'prefill' and ppl['cache_dtype'] == 'SSD scan internal precision' and
             ppl['cache_bytes'] is None and ppl['logits_chunk_tokens'] == 64 and
             ppl['target_tokens'] == 300963 and len(ppl['windows']) == 147, 'Arm numerical path/population differs')
        total_nll = 0.0
        for actual, (start, count, token_sha) in zip(ppl['windows'], expected_windows):
            need((actual['start'], actual['input_tokens'], actual['target_tokens'], actual['token_sha256_int64le']) ==
                 (start, count, count, token_sha), 'Window token/coverage mismatch')
            need(math.isfinite(actual['nll']) and actual['nll'] >= 0, 'Invalid window NLL')
            close(actual['ppl'], math.exp(actual['nll']/count), 'Window PPL arithmetic mismatch')
            total_nll += actual['nll']
        close(ppl['nll'], total_nll, 'Aggregate NLL differs from exact full window sum')
        close(ppl['ppl'], math.exp(total_nll/300963), 'Aggregate PPL arithmetic differs')
        close(receipt['ppl'], ppl['ppl'], 'Arm receipt PPL differs')
        close(report['ppl'][arm], ppl['ppl'], 'Comparison PPL differs')
        rows[arm] = ppl
    left, right = (rows[arm] for arm in ARMS)
    close(report['ppl_relative_change'], right['ppl']/left['ppl']-1, 'PPL relative change differs')
    need(report['matched_windows_improved'] == sum(r['nll'] < l['nll'] for l, r in zip(left['windows'], right['windows'])),
         'Matched improved-window count differs')
    return {'format': FORMAT + '_CPU_AUDIT', 'complete': True, 'passed': True,
            'comparison_sha256': sha(directory / 'comparison.json'), 'pins_sha256': sha(PINS),
            'protocol_sha256': sha(PROTOCOL), 'runner_sha256': sha(RUNNER), 'auditor_sha256': sha(__file__),
            'model_kind': kind, 'split': 'test', 'ppl_execution': 'prefill',
            'windows': 147, 'targets': 300963, 'ppl': report['ppl'],
            'cuda_initialized': False, 'torch_imported': 'torch' in sys.modules,
            'scope': 'Independent saved-token/hash/NLL arithmetic and frozen-checkpoint evidence audit; no GPU inference replay.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    path = args.output_dir / 'cpu_audit_v1.json'
    need(not path.exists(), 'Preserve existing CPU audit')
    result = audit(args.output_dir)
    put(path, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
