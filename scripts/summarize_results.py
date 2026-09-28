#!/usr/bin/env python3
"""Independently audit full W4/Resurface receipts and summarize measured quality.

This CPU-only audit recomputes PPL and recall from raw evaluation rows, verifies
training schedule/provenance, and reads the actual serialized adapter. It does
not replay GPU inference, prove the optimizer executed, or turn reported base
identity checks into an independent post-training hash of all model tensors.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = '47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb'
TOKENIZER_SHA = '5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09'
PROSE_MANIFEST_SHA = 'facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d'
PROSE_CONTENT_SHA = '049dfb6847d778897fd19b67653c3189f29fbc19a36022e8a4801306d08228bc'
ADAPTER_FORMAT = 'MAMBA2_POST_D_RESURFACE_FP16_V1'
MATCH = re.compile(r'(?<!\d)\d{6}(?!\d)')
SHA = re.compile(r'[0-9a-f]{64}')


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    def reject(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    value = json.loads(Path(path).read_text(), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class Audit:
    def __init__(self):
        self.errors = []
        self.checks = 0

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            self.errors.append(label)
        return bool(condition)

    def equal(self, actual, expected, label):
        return self.check(actual == expected, label)

    def close(self, actual, expected, label):
        return self.check(finite(actual) and finite(expected) and
                          math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), label)

    def digest(self, value, label):
        return self.check(isinstance(value, str) and SHA.fullmatch(value) is not None, label)


def audit_ppl(value, label, audit):
    rows = value['windows']
    audit.equal(len(rows), 130, f'{label}: 130 windows')
    audit.equal(value['execution'], 'prefill', f'{label}: declared scan execution')
    audit.equal(value['cache_dtype'], 'SSD scan internal precision', f'{label}: scan precision')
    audit.equal(value['logits_chunk_tokens'], 64, f'{label}: vocabulary chunk size')
    audit.check(value['cache_bytes'] is None, f'{label}: scan has no per-token cache-memory claim')
    signatures = []
    for i, row in enumerate(rows):
        prefix = f'{label}.window[{i}]'
        expected_targets = min(2048, 264764 - 2048 * i)
        audit.equal(row['start'], 2048 * i, prefix + ': contiguous reset-window start')
        audit.equal(row['input_tokens'], expected_targets, prefix + ': input count')
        audit.equal(row['target_tokens'], expected_targets, prefix + ': target count')
        audit.digest(row['token_sha256_int64le'], prefix + ': token hash')
        if audit.check(finite(row['nll']) and row['nll'] >= 0 and expected_targets > 0,
                       prefix + ': finite nonnegative NLL'):
            try:
                audit.close(row['ppl'], math.exp(row['nll'] / expected_targets), prefix + ': PPL from NLL')
            except OverflowError:
                audit.check(False, prefix + ': exponent overflow')
        signatures.append((row['start'], row['target_tokens'], row['token_sha256_int64le']))
    targets = sum(row['target_tokens'] for row in rows)
    total_nll = math.fsum(row['nll'] for row in rows)
    audit.equal(targets, 264764, label + ': full target total')
    audit.equal(value['target_tokens'], targets, label + ': declared target total')
    audit.close(value['nll'], total_nll, label + ': summed raw NLL')
    ppl = math.exp(total_nll / targets)
    audit.close(value['ppl'], ppl, label + ': token-weighted PPL')
    return {'ppl': ppl, 'nll': total_nll, 'target_tokens': targets}, signatures


def audit_mk(value, label, audit, tokenizer=None):
    rows = value['rows']
    audit.equal(len(rows), 768, label + ': 768 raw rows')
    expected_order = [f'resurface-confirm-n{n}-t{t}-s{s}{suffix}'
                      for n in (16, 64) for t in range(3) for s in range(64)
                      for suffix in ('', '-removed')]
    audit.equal([r['id'] for r in rows], expected_order, label + ': exact CONFIRM identity/order')
    cells, correct, signatures = Counter(), Counter(), []
    derived_correct = []
    for i, row in enumerate(rows):
        prefix = f'{label}.row[{i}]'
        expected_condition = 'target_removed' if i % 2 else 'normal'
        audit.equal(row['condition'], expected_condition, prefix + ': paired condition')
        audit.equal(row['N'], 16 if i < 384 else 64, prefix + ': N cell')
        audit.equal(row['template'], (i % 384) // 128, prefix + ': template cell')
        audit.check(isinstance(row['expected'], str) and re.fullmatch(r'\d{6}', row['expected']) is not None,
                    prefix + ': six-digit expected answer')
        audit.digest(row['prompt_token_sha256_int64le'], prefix + ': prompt hash')
        audit.check(type(row['prompt_tokens']) is int and 0 < row['prompt_tokens'] <= 4084,
                    prefix + ': native context budget')
        audit.equal(row['cache_bytes'], 122028032, prefix + ': FP16 cache bytes')
        ids = row['generated_ids']
        valid_ids = isinstance(ids, list) and 1 <= len(ids) <= 12 and all(
            type(t) is int and 0 <= t < 256000 for t in ids)
        audit.check(valid_ids, prefix + ': full-vocabulary generated IDs and length')
        if tokenizer is not None and valid_ids:
            audit.equal(tokenizer.decode(ids), row['output'], prefix + ': decoded generated IDs')
            eos = tokenizer.eos_id()
            if len(ids) < 12:
                audit.equal(ids[-1], eos, prefix + ': short generation ends with EOS')
            audit.check(eos not in ids[:-1], prefix + ': generation stops on first EOS')
        match = MATCH.search(row['output'])
        prediction = match.group(0) if match else None
        is_correct = prediction == row['expected']
        audit.equal(row['prediction'], prediction, prefix + ': first standalone six-digit prediction')
        audit.check(type(row['correct']) is bool and row['correct'] == is_correct,
                    prefix + ': correctness from decoded output')
        key = (row['condition'], row['N'], row['template'])
        cells[key] += 1
        correct[key] += is_correct
        derived_correct.append(is_correct)
        signatures.append(tuple(row[k] for k in ('id', 'condition', 'N', 'template', 'expected',
                                                  'prompt_token_sha256_int64le', 'prompt_tokens')))
    metrics = {}
    for condition in ('normal', 'target_removed'):
        for n in (16, 64):
            for t in range(3):
                audit.equal(cells[condition, n, t], 64, f'{label}: {condition}/N{n}/T{t} count')
        count = sum(v for k, v in cells.items() if k[0] == condition)
        wins = sum(v for k, v in correct.items() if k[0] == condition)
        summary = value['summary'][condition]
        audit.equal(summary['count'], count, f'{label}: {condition} summary count')
        audit.equal(summary['correct'], wins, f'{label}: {condition} summary correct')
        audit.close(summary['accuracy'], wins / count, f'{label}: {condition} summary accuracy')
        metrics[condition] = {'correct': wins, 'count': count, 'accuracy_percent': 100 * wins / count,
            'by_N': {str(n): {'correct': sum(correct[condition, n, t] for t in range(3)),
                             'count': sum(cells[condition, n, t] for t in range(3))} for n in (16, 64)}}
    return metrics, signatures, derived_correct


def audit_evaluation(report, label, audit, tokenizer=None, historical=False):
    expected_format = 'MAMBA2_SOURCE_RESURFACE_EVAL_V1' if historical else 'MAMBA2_W4_RESURFACE_EVAL_V1'
    audit.equal(report['format'], expected_format, label + ': format')
    audit.check(report['complete'] is True and report['smoke'] is False, label + ': completed full evaluation')
    audit.equal(report['split'], 'confirm', label + ': CONFIRM split')
    audit.equal(report['source_checkpoint_sha256'], SOURCE_SHA, label + ': official base hash')
    audit.equal(report['tokenizer_sha256'], TOKENIZER_SHA, label + ': tokenizer hash')
    audit.equal(report['ppl_windows'], 130, label + ': declared windows')
    audit.equal(report['mk_cases'], 768, label + ': declared MK count')
    audit.digest(report['protocol_sha256'], label + ': protocol hash')
    audit.digest(report['data_manifest_sha256'], label + ': data manifest hash')
    audit.equal(report['dataset']['total_tokens'], 264765, label + ': corpus token count')
    audit.equal(report['dataset']['split'], 'validation', label + ': prose split')
    audit.equal(report['dataset']['revision_argument'], 'b08601e04326c79dfdd32d625aee71d232d685c3',
                label + ': pinned WikiText revision')
    audit.equal(report['dataset']['tokenizer_sha256'], TOKENIZER_SHA, label + ': corpus tokenizer')
    metrics, window_sigs, prompt_sigs, derived = {}, {}, {}, {}
    for arm in ('baseline', 'adapter'):
        ppl, window_sigs[arm] = audit_ppl(report[arm + '_ppl'], f'{label}.{arm}_ppl', audit)
        mk, prompt_sigs[arm], derived[arm] = audit_mk(report[arm + '_mk'], f'{label}.{arm}_mk', audit, tokenizer)
        metrics[arm] = {'ppl': ppl['ppl'], 'nll': ppl['nll'], 'target_tokens': ppl['target_tokens'], 'mk': mk}
    audit.equal(window_sigs['baseline'], window_sigs['adapter'], label + ': paired prose identity')
    audit.equal(prompt_sigs['baseline'], prompt_sigs['adapter'], label + ': paired MK identity')
    comparisons = {}
    for parity, condition in enumerate(('normal', 'target_removed')):
        pairs = list(zip(derived['baseline'][parity::2], derived['adapter'][parity::2]))
        counts = Counter(pairs)
        comparison = {'both_correct': counts[True, True], 'both_wrong': counts[False, False],
                      'gained': counts[False, True], 'lost': counts[True, False], 'count': len(pairs),
                      'baseline_correct': sum(a for a, _ in pairs), 'adapter_correct': sum(b for _, b in pairs),
                      'delta_percentage_points': 100 * (counts[False, True] - counts[True, False]) / len(pairs)}
        for key, value in comparison.items():
            if key == 'delta_percentage_points':
                audit.close(report['mk_comparison'][condition][key], value, f'{label}: paired {condition}/{key}')
            else:
                audit.equal(report['mk_comparison'][condition][key], value, f'{label}: paired {condition}/{key}')
        comparisons[condition] = comparison
    delta = 100 * (metrics['adapter']['ppl'] / metrics['baseline']['ppl'] - 1)
    audit.close(report['ppl_delta_percent'], delta, label + ': PPL delta')
    audit.equal(report['restored_probe']['rows'], report['baseline_mk']['rows'][:8], label + ': exact restored probe')
    frozen = report['frozen_base_check']
    audit.check(frozen['identity_version_gradients_unchanged'] is True and frozen['tensors'] == 507
                and frozen['parameters'] == 8236999680, label + ': reported frozen-base inventory')
    return {'arms': metrics, 'ppl_delta_percent': delta, 'mk_comparison': comparisons}, window_sigs['baseline'], prompt_sigs['baseline']


def audit_training(report, prose, audit):
    import torch
    audit.equal(report['format'], 'MAMBA2_W4_RESURFACE_TRAIN_V1', 'training: format')
    audit.check(report['complete'] is True and report['mode'] == 'formal', 'training: completed formal run')
    audit.check(report['zero_adapter_hidden_exact'] is True, 'training: reported exact initial teacher/student probe')
    audit.check(report['teacher_base_parameters_frozen'] is True, 'training: reported frozen teacher')
    audit.equal(report['successful_updates'], 1536, 'training: successful update count')
    binding = report['binding']
    audit.equal(binding['successful_updates'], 1536, 'training binding: final candidate only')
    audit.equal(binding['source_checkpoint_sha256'], SOURCE_SHA, 'training binding: source hash')
    audit.equal(binding['tokenizer_sha256'], TOKENIZER_SHA, 'training binding: tokenizer hash')
    audit.equal(binding['adapter'], ADAPTER_FORMAT, 'training binding: adapter format')
    audit.equal(binding['prose_manifest_sha256'], PROSE_MANIFEST_SHA, 'training binding: prose manifest')
    audit.equal(binding['prose_tokens_int64le_sha256'], PROSE_CONTENT_SHA, 'training binding: prose content')
    for key in ('w4_manifest_sha256', 'train_manifest_sha256', 'prose_tokens_sha256', 'protocol_sha256'):
        audit.digest(binding[key], 'training binding: ' + key)
    schedule = torch.randperm(1536, generator=torch.Generator(device='cpu').manual_seed(2026092803)).tolist()
    prose_order = prose['schedule']
    audit.equal(sorted(prose_order), list(range(448)), 'training: prose permutation')
    history = report['history']
    audit.equal(report['attempts'], len(history), 'training: recorded attempt count')
    audit.check(1536 <= len(history) <= 1544, 'training: attempt budget')
    success, overflow, successful_ids = 0, 0, []
    for i, row in enumerate(history):
        prefix = f'training.attempt[{i + 1}]'
        audit.equal(row['attempt'], i + 1, prefix + ': uninterrupted attempt ordinal')
        if not audit.check(success < 1536, prefix + ': no post-final attempt'):
            break
        entry = schedule[success]
        n, template, sample = (16 if entry < 768 else 64), (entry % 768) // 256, entry % 256
        audit.equal(row['schedule_entry'], entry, prefix + ': frozen numeric permutation')
        audit.equal(row['case_id'], f'resurface-train-n{n}-t{template}-s{sample}', prefix + ': paired TRAIN identity')
        audit.equal(row['prose_window'], prose_order[success % 448], prefix + ': paired prose window')
        audit.equal(row['prose_start'], 512 * ((success // 448) % 4), prefix + ': paired prose segment')
        audit.check(type(row['answer_targets']) is int and 1 <= row['answer_targets'] <= 12,
                    prefix + ': answer target count')
        audit.check(type(row['overflow']) is bool, prefix + ': boolean overflow')
        for key in ('mk_ce', 'prose_ce', 'prose_kl', 'prose_closure', 'seconds', 'loss_scale'):
            audit.check(finite(row[key]), prefix + ': finite ' + key)
        audit.check(row['seconds'] >= 0 and row['loss_scale'] > 0, prefix + ': time/scale range')
        if row['overflow']:
            overflow += 1
            audit.check(row['gradient_norm_before_clip'] is None, prefix + ': no overflow optimizer step')
        else:
            success += 1
            successful_ids.append(row['case_id'])
            audit.check(finite(row['gradient_norm_before_clip']) and row['gradient_norm_before_clip'] >= 0,
                        prefix + ': finite accepted gradient norm')
        audit.equal(row['successful_updates'], success, prefix + ': uninterrupted successful counter')
    audit.equal(success, 1536, 'training: recomputed successful total')
    audit.check(overflow <= 8, 'training: overflow retry budget')
    audit.equal(len(set(successful_ids)), 1536, 'training: 1536 distinct successful examples')
    audit.equal([Path(c['path']).name for c in report['checkpoints']],
                [f'checkpoint_{n:04d}.pt' for n in (384, 768, 1152, 1536)], 'training: fixed checkpoint schedule')
    for checkpoint in report['checkpoints']:
        audit.digest(checkpoint['sha256'], 'training: checkpoint digest')
        audit.check(type(checkpoint['bytes']) is int and checkpoint['bytes'] > 0, 'training: checkpoint byte count')
    frozen = report['frozen_base_check']
    audit.check(frozen['identity_version_gradients_unchanged'] is True and frozen['tensors'] == 507
                and frozen['parameters'] == 8236999680, 'training: reported frozen-base inventory')
    return {'successful_updates': success, 'attempts': len(history), 'overflow_retries': overflow,
            'distinct_successful_examples': len(set(successful_ids)), 'schedule_verified': True}


def audit_adapter(path, training, evaluation, audit):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True)
    audit.equal(payload['format'], ADAPTER_FORMAT, 'adapter: serialized format')
    audit.equal(payload['gate_mode'], 'soft', 'adapter: serialized soft gate')
    audit.equal(payload['variant'], 'post-D native norm-prehook; memoryless cross-head mixing', 'adapter: variant')
    audit.equal(payload['binding'], training['binding'], 'adapter: complete serialized/training binding')
    audit.equal(payload['geometry'], [{'width': 4096, 'heads': 128, 'head_dim': 64}] * 56, 'adapter: 56-layer geometry')
    expected = {f'layer{layer}.{field}': shape for layer in range(56) for field, shape in
                (('V_read', (128, 128)), ('g_read', (128,)), ('router_w', (4096,)), ('router_b', ()))}
    tensors = payload['tensors']
    audit.equal(set(tensors), set(expected), 'adapter: complete tensor inventory')
    hashes, parameters, byte_count = {}, 0, 0
    for key, value in tensors.items():
        valid = isinstance(value, torch.Tensor) and value.dtype == torch.float16
        audit.check(valid, 'adapter: FP16 tensor ' + key)
        if not valid:
            continue
        audit.equal(tuple(value.shape), expected.get(key), 'adapter: shape ' + key)
        audit.check(bool(torch.isfinite(value).all()), 'adapter: finite values ' + key)
        hashes[key] = hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()
        parameters += value.numel()
        byte_count += value.numel() * value.element_size()
    receipt = training['adapter']
    digest = sha_file(path)
    audit.equal(digest, receipt['sha256'], 'adapter: actual file/training SHA')
    audit.equal(digest, evaluation['adapter_sha256'], 'adapter: actual file/evaluation SHA')
    audit.equal(Path(path).stat().st_size, receipt['bytes'], 'adapter: actual serialized bytes')
    audit.equal(hashes, receipt['tensor_sha256'], 'adapter: all serialized tensor hashes')
    audit.equal(parameters, 1154104, 'adapter: parameter count')
    audit.equal(receipt['parameters'], parameters, 'adapter: declared parameter count')
    audit.equal(receipt['payload_bytes'], byte_count, 'adapter: declared payload bytes')
    audit.equal(byte_count, 2308208, 'adapter: expected payload bytes')
    audit.equal(receipt['gate_mode'], 'soft', 'adapter: reported gate')
    audit.check(receipt['roundtrip_bitwise_equal'] is True, 'adapter: reported export roundtrip')
    return {'sha256': digest, 'file_bytes': Path(path).stat().st_size, 'payload_bytes': byte_count,
            'parameters': parameters, 'tensors': len(tensors), 'serialized_binding_verified': True}



def audit_package(manifest, manifest_path, training, evaluation, protocol_sha, audit, package_dir=None):
    audit.equal(manifest['format'], 'mamba2-independent-affine-w4-v1', 'package: independent W4 format')
    audit.equal(manifest['version'], 1, 'package: version')
    audit.check(manifest['complete'] is True, 'package: completed quantization')
    audit.equal(manifest['source_checkpoint_sha256'], SOURCE_SHA, 'package: official source hash')
    audit.equal(manifest['protocol_sha256'], protocol_sha, 'package: protocol hash')
    audit.equal(manifest['license'], 'Apache-2.0', 'package: base weight license')
    audit.equal(manifest['source_repo'], 'nvidia/mamba2-8b-3t-4k', 'package: source repository')
    audit.equal(manifest['source_revision'], 'b915550c63ba9359f88f44d1f6a600d85af27302', 'package: source revision')
    for key, value in (('parameter_count', 8236999680), ('tensor_count', 507),
                       ('w4_tensor_count', 114), ('fp16_tensor_count', 393)):
        audit.equal(manifest[key], value, 'package: ' + key)
    expected_config = {
        'd_model': 4096, 'd_intermediate': 0, 'n_layer': 56, 'vocab_size': 256000,
        'ssm_cfg': {'layer': 'Mamba2', 'd_state': 128, 'd_conv': 4, 'expand': 2,
                    'headdim': 64, 'ngroups': 8, 'chunk_size': 128, 'rmsnorm': True,
                    'norm_before_gate': False, 'use_mem_eff_path': False},
        'rms_norm': True, 'residual_in_fp32': False, 'fused_add_norm': False,
        'pad_vocab_size_multiple': 128, 'tie_embeddings': False}
    audit.equal(manifest['model_config'], expected_config, 'package: pure 56-layer Mamba2 architecture')
    q = manifest['quantization']
    for key, expected in {'method': 'uniform_affine_centered_minmax_weight_mse', 'group_size': 128,
                          'clipping_factors': [1., .99, .98, .97, .96, .95, .94, .92, .90],
                          'scale_dtype': 'float16', 'offset_dtype': 'float16', 'code_bits': 4,
                          'code_order': 'low_nibble_first', 'comparison_dtype': 'float32',
                          'decoded_dtype': 'float16', 'source_reference_dtype': 'float16',
                          'training_data_used': False}.items():
        audit.equal(q[key], expected, 'package: quantization ' + key)
    expected_w4 = {'backbone.embedding.weight': [256000, 4096], 'lm_head.weight': [256000, 4096]}
    for layer in range(56):
        expected_w4[f'backbone.layers.{layer}.mixer.in_proj.weight'] = [18560, 4096]
        expected_w4[f'backbone.layers.{layer}.mixer.out_proj.weight'] = [4096, 8192]
    entries = manifest['tensors']
    audit.equal(len(entries), 507, 'package: tensor inventory count')
    names, kinds, total_parameters, total_bytes, w4_parameters = set(), Counter(), 0, 0, 0
    for name, entry in entries.items():
        prefix = 'package.' + name
        audit.check(isinstance(entry['file'], str) and re.fullmatch(r'[0-9]{4}\.w4bin', entry['file']) is not None
                    and entry['file'] not in names, prefix + ': unique safe filename')
        names.add(entry['file'])
        shape = entry['shape']
        audit.check(isinstance(shape, list) and shape and all(type(n) is int and n > 0 for n in shape),
                    prefix + ': positive tensor shape')
        numel = math.prod(shape)
        audit.equal(entry['numel'], numel, prefix + ': element count')
        total_parameters += numel
        expected_kind = 'w4_affine_f16' if name in expected_w4 else 'fp16'
        audit.equal(entry['kind'], expected_kind, prefix + ': storage kind')
        kinds[entry['kind']] += 1
        if expected_kind == 'w4_affine_f16':
            audit.equal(shape, expected_w4[name], prefix + ': projection/vocabulary shape')
            audit.equal(entry['group_size'], 128, prefix + ': group size')
            payload_bytes = shape[0] * (4 * ((shape[1] + 127) // 128) + (shape[1] + 1) // 2)
            w4_parameters += numel
        else:
            payload_bytes = 2 * numel
        audit.check(type(entry['file_bytes']) is int and entry['file_bytes'] > payload_bytes,
                    prefix + ': packed payload plus header bytes')
        total_bytes += entry['file_bytes']
        audit.digest(entry['sha256'], prefix + ': packed SHA')
        audit.digest(entry['decoded_sha256'], prefix + ': FP16 decoded SHA')
        if package_dir is not None:
            path = package_dir / entry['file']
            audit.check(path.is_file() and not path.is_symlink() and path.resolve().parent == package_dir.resolve(),
                        prefix + ': real local package file')
            audit.equal(path.stat().st_size, entry['file_bytes'], prefix + ': actual file bytes')
            audit.equal(sha_file(path), entry['sha256'], prefix + ': actual packed bytes hash')
            with path.open('rb') as stream:
                magic, version, header_length = struct.unpack('<8sII', stream.read(16))
                audit.equal(magic, b'M2W4PK01', prefix + ': file magic')
                audit.equal(version, 1, prefix + ': file version')
                if not 0 < header_length <= 65536:
                    raise ValueError(prefix + ': invalid header length')
                header = json.loads(stream.read(header_length))
            audit.equal(header['kind'], entry['kind'], prefix + ': actual header storage kind')
            audit.equal(header['shape'], shape, prefix + ': actual header shape')
            audit.equal(16 + header_length + payload_bytes, entry['file_bytes'], prefix + ': complete payload length')
    audit.equal(set(name for name, entry in entries.items() if entry['kind'] == 'w4_affine_f16'),
                set(expected_w4), 'package: all 114 expected large matrices')
    audit.equal(kinds, Counter({'w4_affine_f16': 114, 'fp16': 393}), 'package: exact storage coverage')
    audit.equal(total_parameters, 8236999680, 'package: summed parameters')
    audit.equal(manifest['tensor_bytes'], total_bytes, 'package: summed packed file bytes')
    manifest_sha = sha_file(manifest_path)
    for label, report in (('training', training), ('evaluation', evaluation)):
        receipt = report['w4_package']
        for key in ('format', 'source_checkpoint_sha256', 'parameter_count', 'tensor_count',
                    'w4_tensor_count', 'fp16_tensor_count', 'tensor_bytes'):
            audit.equal(receipt[key], manifest[key], label + ': loaded package receipt ' + key)
        audit.equal(receipt['manifest_sha256'], manifest_sha, label + ': loaded package manifest hash')
        audit.equal(receipt['resident_weight_dtype'], 'float16', label + ': decoded reference precision')
        audit.check(receipt['packed_resident_kernel'] is False, label + ': packed runtime non-claim')
        audit.check(receipt['file_hashes_verified'] is True and receipt['decoded_hashes_verified'] is True,
                    label + ': loader reported packed and decoded hashes verified')
    return {'packed_tensor_file_bytes': total_bytes, 'manifest_bytes': Path(manifest_path).stat().st_size,
            'weights_plus_manifest_bytes': total_bytes + Path(manifest_path).stat().st_size,
            'effective_tensor_file_bits_per_all_parameters': 8 * total_bytes / total_parameters,
            'parameters': total_parameters, 'w4_parameters': w4_parameters,
            'fp16_parameters': total_parameters - w4_parameters,
            'tensor_files': len(entries), 'packed_file_hashes_independently_rechecked': package_dir is not None,
            'decoded_tensor_hashes_independently_rechecked': False,
            'reference_resident_weight_bytes': 2 * total_parameters,
            'note': 'Tensor files include headers. Added license/docs/tokenizer/adapter bytes are not included in base totals.'}


def markdown(result):
    lines = ['# W4A16 Resurface result audit', '',
             f"Integrity audit: **{'PASS' if result['integrity_pass'] else 'FAIL'}** "
             f"({result['checks']} checks, {len(result['errors'])} errors).", '']
    if 'evaluation' in result:
        lines += ['| Arm | PPL | Normal recall | Target removed |', '|---|---:|---:|---:|']
        groups = [('W4', result['evaluation'])]
        if 'historical_source' in result:
            groups.append(('Historical source FP16', result['historical_source']))
        for group, value in groups:
            for arm, data in value['arms'].items():
                normal, removed = data['mk']['normal'], data['mk']['target_removed']
                suffix = 'base' if arm == 'baseline' else '+ Resurface'
                lines.append(f"| {group} {suffix} | {data['ppl']:.6f} | "
                             f"{normal['correct']}/{normal['count']} ({normal['accuracy_percent']:.2f}%) | "
                             f"{removed['correct']}/{removed['count']} |")
        lines += ['', 'Full WikiText-2 validation: 130 reset windows, 264,764 next-token targets. '
                  'MK: 384 normal plus 384 target-removed cases; previously observed template family and instances.']
    if result['errors']:
        lines += ['', '## Audit errors', ''] + ['- ' + value for value in result['errors']]
    lines += ['', '## Scope', ''] + ['- ' + value for value in result['scope']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--package-manifest', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--package-dir', type=Path, help='Optional actual packed tensor directory for independent file hashing')
    parser.add_argument('--historical', type=Path)
    parser.add_argument('--train-manifest', type=Path)
    parser.add_argument('--eval-manifest', type=Path)
    parser.add_argument('--prose-manifest', type=Path, default=ROOT / 'docs' / 'prose_train_manifest.json')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'docs' / 'PROTOCOL.md')
    parser.add_argument('--tokenizer', type=Path, help='Optional pinned SentencePiece file for generated-ID decoding')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--markdown', type=Path)
    args = parser.parse_args()
    if args.markdown and args.output.resolve() == args.markdown.resolve():
        parser.error('JSON and Markdown outputs must be different files')
    for path in (args.output, args.markdown):
        if path and path.exists():
            parser.error(f'Preserve existing audit: {path}')
    audit = Audit()
    result = {'format': 'MAMBA2_W4_RESURFACE_SEMANTIC_AUDIT_V1', 'complete': False,
              'integrity_pass': False, 'inputs': {}, 'scope': [
                  'CPU semantic audit of supplied reports and actual serialized adapter; no GPU replay.',
                  'PPL is recomputed from raw window NLL and target counts; recall from recorded generated text.',
                  'Training schedule and reported frozen-base status are checked; optimizer execution and all base bytes are not independently replayed.',
                  'Previously observed CONFIRM instances and WikiText validation; no untouched-generalization claim.',
                  'Packed W4 file size is separate from the decoded FP16 reference runtime memory.']}
    try:
        paths = {key: getattr(args, key) for key in ('evaluation', 'training', 'package_manifest', 'adapter',
                 'historical', 'train_manifest', 'eval_manifest', 'prose_manifest', 'protocol', 'tokenizer')}
        for key, path in paths.items():
            if path is not None:
                result['inputs'][key] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha_file(path)}
        evaluation, training, package, prose = [load_json(path) for path in
            (args.evaluation, args.training, args.package_manifest, args.prose_manifest)]
        audit.equal(sha_file(args.prose_manifest), PROSE_MANIFEST_SHA, 'inputs: pinned prose selection manifest')
        audit.check(package['complete'] is True, 'package: complete manifest')
        audit.equal(package['source_checkpoint_sha256'], SOURCE_SHA, 'package: original NVIDIA source SHA')
        package_sha = sha_file(args.package_manifest)
        audit.equal(evaluation['w4_manifest_sha256'], package_sha, 'binding: evaluation/package manifest')
        audit.equal(training['binding']['w4_manifest_sha256'], package_sha, 'binding: training/package manifest')
        protocol_sha = sha_file(args.protocol)
        audit.equal(evaluation['protocol_sha256'], protocol_sha, 'binding: evaluation/current protocol')
        audit.equal(training['binding']['protocol_sha256'], protocol_sha, 'binding: training/current protocol')
        if args.train_manifest:
            audit.equal(training['binding']['train_manifest_sha256'], sha_file(args.train_manifest), 'binding: TRAIN manifest')
        if args.eval_manifest:
            audit.equal(evaluation['data_manifest_sha256'], sha_file(args.eval_manifest), 'binding: CONFIRM manifest')
        tokenizer = None
        if args.tokenizer:
            audit.equal(sha_file(args.tokenizer), TOKENIZER_SHA, 'inputs: actual tokenizer hash')
            import sentencepiece as spm
            tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
        result['package'] = audit_package(package, args.package_manifest, training, evaluation, protocol_sha, audit, args.package_dir)
        result['training'] = audit_training(training, prose, audit)
        result['adapter'] = audit_adapter(args.adapter, training, evaluation, audit)
        result['evaluation'], windows, prompts = audit_evaluation(evaluation, 'W4', audit, tokenizer)
        audit.check(finite(training['finished_unix']) and finite(evaluation['started_unix']) and
                    training['finished_unix'] <= evaluation['started_unix'], 'timeline: evaluation follows final training')
        if args.historical:
            historical = load_json(args.historical)
            result['historical_source'], old_windows, old_prompts = audit_evaluation(
                historical, 'historical source', audit, tokenizer, historical=True)
            audit.equal(windows, old_windows, 'historical comparison: all 130 window hashes/targets/starts')
            audit.equal(prompts, old_prompts, 'historical comparison: all 768 prompt hashes/answers/geometry')
            for key in ('text_sha256', 'token_stream_sha256_int64le', 'total_tokens', 'tokenizer_sha256'):
                audit.equal(evaluation['dataset'][key], historical['dataset'][key], 'historical comparison: corpus ' + key)
            result['historical_comparison_scope'] = 'Matched recorded prompts/windows; historical source execution was not rerun.'
            result['w4_vs_historical_source'] = {}
            for arm in ('baseline', 'adapter'):
                new = result['evaluation']['arms'][arm]
                old = result['historical_source']['arms'][arm]
                result['w4_vs_historical_source'][arm] = {
                    'ppl_delta_percent': 100 * (new['ppl'] / old['ppl'] - 1),
                    'normal_recall_delta_percentage_points': new['mk']['normal']['accuracy_percent'] - old['mk']['normal']['accuracy_percent']} 
        base = result['evaluation']['arms']['baseline']
        adapted = result['evaluation']['arms']['adapter']
        result['quality_observations'] = {
            'adapter_ppl_not_worse_than_w4_base': adapted['ppl'] <= base['ppl'],
            'adapter_normal_recall_improved': adapted['mk']['normal']['correct'] > base['mk']['normal']['correct'],
            'adapter_target_removed_correct': adapted['mk']['target_removed']['correct']}
        result['generated_ids_decoded_with_pinned_tokenizer'] = tokenizer is not None
        if args.package_dir is None:
            result['scope'].append('Packed file identity is bound through the manifest and loader receipts; actual packed files were not rehashed without --package-dir.')
        if tokenizer is None:
            result['scope'].append('Generated-ID range/length checked; ID-to-text decoding was not checked without --tokenizer.')
        result['complete'] = True
    except Exception as error:
        audit.check(False, f'Audit could not finish: {type(error).__name__}: {error}')
    result.update(integrity_pass=result['complete'] and not audit.errors, checks=audit.checks, errors=audit.errors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        with args.markdown.open('x') as stream:
            stream.write(markdown(result))
    print(json.dumps({'integrity_pass': result['integrity_pass'], 'checks': result['checks'],
                      'errors': result['errors'], 'output': str(args.output)}))
    return 0 if result['integrity_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
