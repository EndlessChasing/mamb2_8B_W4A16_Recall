#!/usr/bin/env python3
"""Package already verified weights for Hub and split GitHub release delivery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'mamb2_8B_W4A16_Recall-v0.1.0'
MANIFEST_SHA = '3add3f79f19d2da181c700680500390f773a47b2785d8f6e0ccaaf2ddd7bbc05'
ADAPTER_SHA = '6d38c24a102cbf83c5f1dfe4362a0b5c5e779e793a27bccd4e592ff6c9de7ca1'
TOKENIZER_SHA = '5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09'
TOKENIZER = 'mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


class SplitWriter:
    def __init__(self, directory):
        self.directory = directory
        self.files = []
        self.stream = None
        self.size = 0

    def write(self, data):
        original = len(data)
        while data:
            if self.stream is None or self.size == 1_500_000_000:
                if self.stream:
                    self.stream.close()
                path = self.directory / f'{PREFIX}.tar.part-{len(self.files):02d}'
                self.files.append(path)
                self.stream = path.open('xb')
                self.size = 0
            count = min(len(data), 1_500_000_000 - self.size)
            self.stream.write(data[:count])
            data = data[count:]
            self.size += count
        return original

    def close(self):
        if self.stream:
            self.stream.close()


class JoinedReader:
    def __init__(self, files):
        self.files = iter(files)
        self.stream = None

    def read(self, count):
        result = bytearray()
        while len(result) < count:
            if self.stream is None:
                path = next(self.files, None)
                if path is None:
                    break
                self.stream = path.open('rb')
            block = self.stream.read(count - len(result))
            result.extend(block)
            if not block:
                self.stream.close()
                self.stream = None
        return bytes(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve the existing release bundle')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip():
        raise RuntimeError('Commit release sources before packaging')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    base = ROOT / 'artifacts/w4_base_v1'
    manifest = json.loads((base / 'manifest.json').read_text())
    assert sha(base / 'manifest.json') == MANIFEST_SHA
    assert sha(ROOT / 'pretrained/w4_resurface_v1/adapter_fp16.pt') == ADAPTER_SHA
    assert sha(ROOT / 'artifacts/release_source' / TOKENIZER) == TOKENIZER_SHA
    for name, digest in manifest['code_sha256'].items():
        assert sha(ROOT / name) == digest, name
    audit = json.loads((ROOT / 'reports/semantic_audit_v1.json').read_text())
    assert audit['integrity_pass'] and audit['checks'] == 75146 and not audit['errors']
    bundle = args.output / 'hf_bundle'
    assets = args.output / 'github_assets'
    (bundle / 'w4_base').mkdir(parents=True)
    assets.mkdir()
    for entry in manifest['tensors'].values():
        source = base / entry['file']
        assert source.is_file() and not source.is_symlink()
        assert source.stat().st_size == entry['file_bytes'] and sha(source) == entry['sha256']
        os.link(source, bundle / 'w4_base' / source.name)
    copy(base / 'manifest.json', bundle / 'w4_base/manifest.json')
    copy(ROOT / 'pretrained/w4_resurface_v1/adapter_fp16.pt', bundle / 'adapter/adapter_fp16.pt')
    copy(ROOT / 'pretrained/w4_resurface_v1/LICENSE.txt', bundle / 'adapter/LICENSE.txt')
    notice = (ROOT / 'pretrained/w4_resurface_v1/NOTICE.md').read_text().replace('../../docs/', '../docs/')
    (bundle / 'adapter/NOTICE.md').write_text(notice)
    copy(ROOT / 'artifacts/release_source' / TOKENIZER, bundle / 'tokenizer' / TOKENIZER)
    for directory in ('mamba2_recall', 'scripts'):
        for source in sorted((ROOT / directory).glob('*.py')):
            copy(source, bundle / 'code' / directory / source.name)
    for name in ('pyproject.toml', 'LICENSE'):
        copy(ROOT / name, bundle / 'code' / name)
    for name in ('PROTOCOL.md', 'prose_train_manifest.json'):
        copy(ROOT / 'docs' / name, bundle / 'code/docs' / name)
    for directory in ('docs', 'reports'):
        for source in sorted((ROOT / directory).iterdir()):
            if source.is_file():
                copy(source, bundle / directory / source.name)
    copy(ROOT / 'docs/HF_MODEL_CARD.md', bundle / 'README.md')
    copy(ROOT / 'docs/WEIGHTS_LICENSE.txt', bundle / 'LICENSE')
    (bundle / 'NOTICE').write_text(
        'Mamb2_8B_W4A16_Recall v0.1.0\n\n'
        'Original model and tokenizer: NVIDIA nvidia/mamba2-8b-3t-4k, Apache-2.0.\n'
        'Pinned original revision: b915550c63ba9359f88f44d1f6a600d85af27302.\n'
        'Copyright 2026 EndlessChasing for project contributions.\n'
        'Modifications: independent affine group-128 INT4 quantization of 114 large\n'
        'matrices; 393 small tensors retained in FP16; separately trained post-D\n'
        'Resurface-inspired adapter. Quantized weights and adapter: Apache-2.0.\n'
        'Retain LICENSE, this notice, docs/WEIGHTS_NOTICE.md, and adapter/NOTICE.md.\n'
        'Implementation code: GPL-3.0 under code/LICENSE. Dependencies retain their licenses.\n'
        'Upstream authors do not endorse these modifications. No Quamba artifacts are included.\n'
        'See README.md and docs/WEIGHTS_NOTICE.md for full provenance and references.\n')
    inventory = {}
    for path in sorted(bundle.rglob('*')):
        if path.is_file():
            inventory[path.relative_to(bundle).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    receipt = {'format': 'MAMBA2_W4_RELEASE_BUNDLE_V1', 'version': 'v0.1.0',
               'github_source_commit': revision, 'base_manifest_sha256': MANIFEST_SHA,
               'adapter_sha256': ADAPTER_SHA, 'tokenizer_sha256': TOKENIZER_SHA,
               'core_base_bytes': 4381415300, 'adapter_bytes': 2374271,
               'payload_files': inventory,
               'scope': 'Inventory excludes RELEASE_MANIFEST.json and SHA256SUMS themselves.'}
    (bundle / 'RELEASE_MANIFEST.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    inventory['RELEASE_MANIFEST.json'] = {'bytes': (bundle / 'RELEASE_MANIFEST.json').stat().st_size,
                                         'sha256': sha(bundle / 'RELEASE_MANIFEST.json')}
    (bundle / 'SHA256SUMS').write_text(''.join(f"{entry['sha256']}  {name}\n" for name, entry in sorted(inventory.items())))
    inventory['SHA256SUMS'] = {'bytes': (bundle / 'SHA256SUMS').stat().st_size, 'sha256': sha(bundle / 'SHA256SUMS')}
    writer = SplitWriter(assets)
    with tarfile.open(fileobj=writer, mode='w|', format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(inventory):
            path = bundle / name
            info = tarfile.TarInfo(f'{PREFIX}/{name}')
            info.size = path.stat().st_size
            info.mode = 0o644
            info.mtime = 1790553600
            with path.open('rb') as stream:
                archive.addfile(info, stream)
    writer.close()
    assert len(writer.files) == 3
    seen = set()
    with tarfile.open(fileobj=JoinedReader(writer.files), mode='r|') as archive:
        for member in archive:
            assert member.isfile() and member.name.startswith(PREFIX + '/')
            name = member.name[len(PREFIX)+1:]
            assert name in inventory and name not in seen
            digest = hashlib.sha256()
            stream = archive.extractfile(member)
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
            assert member.size == inventory[name]['bytes'] and digest.hexdigest() == inventory[name]['sha256'], name
            seen.add(name)
    assert seen == set(inventory)
    (assets / 'release-assets.sha256').write_text(''.join(f'{sha(path)}  {path.name}\n' for path in writer.files))
    # These convenient small assets are also present inside the complete archive.
    copy(bundle / 'adapter/adapter_fp16.pt', assets / 'adapter_fp16.pt')
    copy(bundle / 'w4_base/manifest.json', assets / 'w4_manifest.json')
    copy(bundle / 'RELEASE_MANIFEST.json', assets / 'RELEASE_MANIFEST.json')
    (assets / 'SHA256SUMS').write_text(''.join(f'{sha(path)}  {path.name}\n' for path in sorted(assets.iterdir())))
    print(json.dumps({'complete': True, 'bundle_files': len(inventory),
                      'bundle_bytes': sum(x['bytes'] for x in inventory.values()),
                      'archive_roundtrip_files_verified': len(seen), 'github_source_commit': revision,
                      'assets': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in assets.iterdir()}}, indent=2))


if __name__ == '__main__':
    main()
