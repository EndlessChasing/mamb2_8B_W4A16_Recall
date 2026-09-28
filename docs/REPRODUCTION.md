# Reproduction

For inference with the complete published checkpoint, use the
[Hugging Face download and generation instructions](HF_MODEL_CARD.md#download-and-generate).
That bundle includes the packed base, adapter and tokenizer and does not require
the original checkpoint or another quantization run. The commands below reproduce
the experiment from the original source in a source-repository checkout.

Run on a CUDA Linux machine. The measured environment is recorded in
[`reports/environment_v1.json`](../reports/environment_v1.json): Python 3.10.12,
PyTorch 2.11.0+cu128, mamba-ssm 2.3.2.post1, Triton 3.6.0, NumPy 1.26.4,
datasets 4.8.5 and sentencepiece 0.2.1. Native Mamba CUDA extensions must work
in that environment. Install the project with `python -m pip install -e .`.
This does not itself install the separate native `mamba-ssm` extension.

The quality reference expands both the teacher and student to FP16. Plan for
roughly 40GB GPU memory for this training recipe. The measured training peak
allocated GPU memory was 34,596,873,216 bytes (about 34.60 GB). Packed-file size is not this memory
requirement. Existing downloaded source weights can be reused in place.

## Download the original licensed source

```bash
hf download nvidia/mamba2-8b-3t-4k \
  --revision b915550c63ba9359f88f44d1f6a600d85af27302 \
  --local-dir models/source
```

The loader verifies the original checkpoint and tokenizer SHA256. It uses a
restricted weights-only loader with inspected passive Megatron metadata types.
No Quamba package or quantized checkpoint is required.

## Prepare the fixed data

```bash
CUDA_VISIBLE_DEVICES= python scripts/prepare_data.py \
  --split train --source-dir models/source
CUDA_VISIBLE_DEVICES= python scripts/prepare_data.py \
  --split confirm --source-dir models/source
CUDA_VISIBLE_DEVICES= python scripts/prepare_prose.py \
  --source-dir models/source --out training_data/prose/training_tokens.pt
```

Numeric data are generated from fixed seeds. The manifest includes the Python
version, so the runner computes its actual hash instead of assuming that a
manifest from a different Python build has identical bytes. The loader checks
all generated content. Prose tokens must match the pinned content hash; the
Torch container-byte identity is recorded separately.

## Quantize, train and evaluate

Use a fresh checkout for a complete run, after preparing the fixed data above.
Its `artifacts/`, `logs/` and `reports/w4_resurface_v1_confirm_full.json` output
paths must be unused. Published reference artifacts live separately under
`pretrained/` and `reports/w4_resurface_v1_confirm_full_reference.json`, so they
do not occupy these run outputs.

```bash
CUDA_VISIBLE_DEVICES= python scripts/check_w4.py --self-test
python scripts/run_experiment.py \
  --source-dir models/source \
  --prose-tokens training_data/prose/training_tokens.pt
```

This sequential runner writes stage logs under `logs/`, runs quantization,
discards one training-smoke update, then runs 1,536 successful updates and
full paired PPL/MK evaluation. It stops on any stage error. Existing output
files are preserved; do not rerun it over a completed experiment.

For individual quantization instead of the complete runner, use the following
commands. Do not also run this first into the complete runner's output path.
Explicit resume verifies files from an interrupted quantization:

```bash
python scripts/quantize_w4.py \
  --source models/source --output artifacts/w4_base_v1
# After an interrupted quantization only:
python scripts/quantize_w4.py \
  --source models/source --output artifacts/w4_base_v1 --resume
```

The weight package is standalone for loading; original source weights are not
needed at inference time. The pinned tokenizer remains necessary. Training and
evaluation verify the actual W4 manifest SHA and actual serialized adapter.

## Independently audit the saved result

After all stages of your new run complete, write the audit to fresh output
paths, preserving the repository's published reference reports:

```bash
CUDA_VISIBLE_DEVICES= python scripts/summarize_results.py \
  --evaluation reports/w4_resurface_v1_confirm_full.json \
  --training artifacts/w4_resurface_v1/report.json \
  --package-manifest artifacts/w4_base_v1/manifest.json \
  --package-dir artifacts/w4_base_v1 \
  --adapter artifacts/w4_resurface_v1/adapter_fp16.pt \
  --train-manifest training_data/numeric_v1/train/manifest.json \
  --eval-manifest training_data/numeric_v1/confirm/manifest.json \
  --tokenizer models/source/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
  --output reports/semantic_audit_reproduced.json \
  --markdown docs/RESULTS_REPRODUCED.md
```

An optional `--historical PATH` accepts the earlier source FP16 paired report.
The audit requires identical recorded window/prompt hashes before comparing
metrics; it does not rerun the historical GPU evaluation.

## Use the included pretrained adapter

The independently trained, Apache-2.0 adapter is included at
[`pretrained/w4_resurface_v1/adapter_fp16.pt`](../pretrained/w4_resurface_v1/adapter_fp16.pt).
To use it without training, download the original source as above and regenerate
its exact W4 base in a separate directory. Use the unmodified quantizer, runtime
and frozen protocol from this repository; their hashes are recorded in
[`reports/w4_base_v1_manifest.json`](../reports/w4_base_v1_manifest.json).
Cross-environment bitwise reproduction has not been established. The published
adapter requires the exact recorded package identity.

```bash
python scripts/quantize_w4.py \
  --source models/source --output artifacts/w4_pretrained_base_v1
python - <<'PY'
from hashlib import sha256
from pathlib import Path

expected = {
    'artifacts/w4_pretrained_base_v1/manifest.json':
        '3add3f79f19d2da181c700680500390f773a47b2785d8f6e0ccaaf2ddd7bbc05',
    'pretrained/w4_resurface_v1/adapter_fp16.pt':
        '6d38c24a102cbf83c5f1dfe4362a0b5c5e779e793a27bccd4e592ff6c9de7ca1',
}
for filename, required in expected.items():
    actual = sha256(Path(filename).read_bytes()).hexdigest()
    if actual != required:
        raise SystemExit(f'STOP: {filename} SHA256 differs: {actual}')
print('Exact pretrained base manifest and adapter identities verified.')
PY
```

Stop if either check fails. Do not edit the manifest, substitute the reference
manifest or rebind the adapter to bypass a mismatch. `generate.py` also rejects
an adapter whose recorded base-manifest or tokenizer identity differs, and the
base loader verifies the actual tensor files against the manifest.

```bash
python scripts/generate.py \
  --w4-dir artifacts/w4_pretrained_base_v1 \
  --tokenizer models/source/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
  --adapter pretrained/w4_resurface_v1/adapter_fp16.pt \
  --prompt 'The key idea of a state space model is' --max-new-tokens 64
```

## Generate with your newly trained adapter

After successful training, use the standalone W4 package, tokenizer and adapter:

```bash
python scripts/generate.py \
  --w4-dir artifacts/w4_base_v1 \
  --tokenizer models/source/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
  --adapter artifacts/w4_resurface_v1/adapter_fp16.pt \
  --prompt 'The key idea of a state space model is' --max-new-tokens 64
```

Omit `--adapter` for the quantized baseline. The helper verifies the adapter's
W4-package and tokenizer hashes. This is a base completion model, not an
instruction/chat model. Its numerical reference uses expanded FP16 GPU weights.

For standalone redistribution, include the outer license and provenance files
described in [WEIGHTS_NOTICE.md](WEIGHTS_NOTICE.md). Do not alter the bound W4
core package to add them.
