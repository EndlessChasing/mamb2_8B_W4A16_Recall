# Frozen FP16 and INT4 Recall: official WikiText-2 test PPL

This is an additional evaluation of the previously published checkpoints. It
does not train, select, rescale, requantize, or alter either checkpoint.

## Checkpoints and inference path

`RECALL_NATIVE_PREFILL_WT2_TEST_V1_PINS.json` fixes the original FP16 Recall and
independent affine group128 INT4 W4A16 Recall adapter files, their training
bindings, the published reference/training reports, original protocol, and all
runtime modules used here. Their previous 130-window PPL results used WikiText-2
**validation**. Both releases used native parallel SSD **prefill**, with internal
scan precision; they did not round recurrent states after every token. This
runner imports the original released `evaluation.evaluate_ppl` unchanged and
calls it explicitly with `execution="prefill", logits_chunk=64`.

The FP16 model is the pinned NVIDIA BF16 source cast to FP16 by its original
loader. The W4 model is decoded from the already published INT4 package by its
original loader; no new quantization occurs. Both remain frozen. These native
prefill baselines cannot be substituted with a per-token S16 baseline measured
by the later StateQuant runtime. This protocol measures each baseline directly.

## Population and arithmetic

- Dataset: `Salesforce/wikitext`, `wikitext-2-raw-v1`, revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`, official **test** split.
- Original join: two newlines between dataset rows; original NVIDIA
  SentencePiece, no automatically inserted BOS or EOS. The pinned test token
  stream SHA-256 is
  `5b82bd46e833e77fcfc0af62bafeaac62e70e68cfdf214d375f0b7b132d4b608`.
- 300,964 token IDs, 300,963 next-token targets, 147 windows. Each full window
  scores 2,048 targets; the final partial window is included. Windows reset
  the native model state and cover each target exactly once.
- Sum next-token cross entropy in the original 64-position FP32 logits chunks;
  compute `exp(total_nll / total_targets)`. Do not average window PPLs.
- Run adapter off, then the fixed exported FP16 adapter on. Each arm uses the
  same complete population. Record per-window NLL and token SHA-256. No smoke,
  truncation, or test-dependent selection option exists.

## Backend and integrity

Retain the original numerical settings: eight CPU threads, matmul TF32 disabled,
float32 matmul precision `highest`. Require the original recorded PyTorch,
Mamba, Triton, NumPy, datasets, SentencePiece versions and CUDA runtime. Record
the remaining backend flags and environment. Do not replace the old parallel
scan with a later singleton recurrent backend policy.

The runner binds its own source, this protocol, and the pins file in every
report. It verifies the fixed published adapter and its individual tensors.
It derives an FP16 weight ledger from the verified BF16 source in bounded CPU
chunks, or from the pinned W4 manifest's decoded hashes. It verifies actual
loaded FP16 contents before and after both arms, original parameter
identity/version/gradient status after each arm, and actual loaded adapter
contents. A fixed first-128-token native hidden-state probe must be bitwise
identical after adapter removal. All original hooks and forward methods must
also be restored. Raw little-endian int64 tokens, expected weight ledger, full
arm reports, and checksums are saved into a fresh output directory.

The independent CPU auditor reconstructs every window from the saved raw token
file and verifies its exact identity, full target coverage, aggregate NLL/PPL,
off/on pairing, all pinned code and checkpoint provenance, weight/adapter
ledgers, restoration evidence, and report hashes. It does not rerun inference
or claim an independent GPU numerical reproduction.

## Interpretation

Label the new result **official WikiText-2 test PPL, native parallel SSD
prefill**; keep the prior numbers explicitly labeled validation. The
checkpoints were frozen before this additional test and are not modified based
on it. Project test text was used in earlier 2.7B work. Base pretraining
contamination and cross-split duplicate text are not audited. These results do
not constitute an untouched generalization test or a new recall score. Existing
MK scores retain their original confirmation-population and inference labels.

## Commands

Run `scripts/run_recall_native_prefill_wt2_test_v1.py` from the research repo in
two separate Python processes, with `--model-kind fp16` or `w4`, the respective
`--release-root`, original `--source-dir` (also supplies the tokenizer), and a
fresh `--out-dir`. W4 additionally requires the frozen `--w4-dir`. A read-only
`--preflight-only` validates frozen files, adapter, and W4 manifest without
importing PyTorch or initializing CUDA. Run
`scripts/audit_recall_native_prefill_wt2_test_v1.py --output-dir ...` afterward.
