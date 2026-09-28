# Reproduction

Run on a CUDA Linux machine. The measured environment is recorded in
[`reports/environment_v1.json`](../reports/environment_v1.json): Python 3.10.12,
PyTorch 2.11.0+cu128, mamba-ssm 2.3.2.post1, Triton 3.6.0, NumPy 1.26.4,
datasets 4.8.5 and sentencepiece 0.2.1. Native Mamba CUDA extensions must work
in that environment. Install the project with `python -m pip install -e .`.
This does not itself install the separate native `mamba-ssm` extension.

The quality reference expands both the teacher and student to FP16. Plan for
roughly 40GB GPU memory for this training recipe; the actual measured peak
belongs in the completed training report. Packed-file size is not this memory
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

Use a fresh checkout/output directory for a complete run:

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

Individual quantization, with interrupted-file verification on explicit resume:

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

After all stages complete:

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
  --output reports/semantic_audit_v1.json --markdown docs/RESULTS.md
```

An optional `--historical PATH` accepts the earlier source FP16 paired report.
The audit requires identical recorded window/prompt hashes before comparing
metrics; it does not rerun the historical GPU evaluation.

## Generate with the final adapter

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
