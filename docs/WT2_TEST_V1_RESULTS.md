# Official WikiText-2 test: Mamb2_8B_W4A16_Recall

The previously published checkpoint is unchanged. This additional evaluation
uses the official **test** split and the release's original **native parallel
SSD prefill** PPL calculation; state is not rounded after each token.

| Fixed model | Official WT2 test PPL |
| --- | ---: |
| Independent INT4 W4A16, adapter off | 7.9076894253529275 |
| Independent INT4 W4A16 + published Resurface adapter | **7.51706425275758** |

Adapter-relative PPL change: **-4.939814%**.
All **147 reset windows / 300,963 targets** are scored, including the final
partial window. Both arms use identical windows, original NVIDIA SentencePiece,
no automatically added BOS/EOS, zero initial state per window, and 64-position
FP32-logit scoring chunks. The original loader, weight decoding and PPL function
are unchanged.

## Historical validation and recall

| Original published arm | WT2 validation PPL | Synthetic CONFIRM normal MK | Target removed |
| --- | ---: | ---: | ---: |
| Independent INT4 W4A16, adapter off | 8.012095592914347 | 141/384 | 0/384 |
| Independent INT4 W4A16 + Resurface | 7.614046815850065 | 361/384 | 0/384 |

The old PPL population is **130 validation windows / 264,764 targets**.
MK remains the original **384 normal synthetic CONFIRM prompts plus 384
target-removed controls**, with a previously observed template family.
It uses native prefill followed by recurrent decoding. This update does
not rerun MK or relabel CONFIRM as a new MK test set.

## Integrity and scope

The fixed serialized adapter and original runtime match the published hashes.
No training or checkpoint selection uses this test. All 507 loaded FP16 base
tensors were checked against independent source/package hashes before and
after evaluation; actual loaded adapter tensors match its 224-tensor ledger.
Removing the adapter restores the native 128-token hidden probe bitwise and
restores native methods/hooks. An independent CPU evidence audit checks
saved-token identities, full coverage, NLL/PPL arithmetic, paired populations,
frozen bytes and restoration receipts; it does not rerun GPU inference.

Project test text was seen in earlier 2.7B experiments. Base pretraining
contamination and cross-split duplicate text are not audited. This additional
official test evaluation is not an untouched generalization claim.

Dataset revision: b08601e04326c79dfdd32d625aee71d232d685c3.
Test token stream SHA-256:
5b82bd46e833e77fcfc0af62bafeaac62e70e68cfdf214d375f0b7b132d4b608.
Adapter SHA-256: 6d38c24a102cbf83c5f1dfe4362a0b5c5e779e793a27bccd4e592ff6c9de7ca1.
Original HF checkpoint revision: e1baffa7aa680f3dc26d7d7df2104756248d1a24.

Evidence: [paired comparison](../reports/wt2_test_v1/comparison.json),
[adapter-off windows](../reports/wt2_test_v1/without_resurface.json),
[adapter-on windows](../reports/wt2_test_v1/resurface.json),
and [CPU audit](../reports/wt2_test_v1/cpu_audit_v1.json).
See the [frozen test protocol](RECALL_NATIVE_PREFILL_WT2_TEST_V1_PROTOCOL.md)
and [checkpoint/code pins](RECALL_NATIVE_PREFILL_WT2_TEST_V1_PINS.json).

## Reproduce official test PPL

Use the recorded original CUDA environment. A GitHub checkout already has the
original runtime layout. For a downloaded Hugging Face bundle, assemble the
small frozen runtime first; no packed weights or source checkpoint are copied.

```bash
python scripts/prepare_recall_wt2_test_inputs_v1.py runtime \
  --model-kind w4 --download-root . --out-dir test_runtime
python scripts/run_recall_native_prefill_wt2_test_v1.py \
  --model-kind w4 --release-root test_runtime \
  --source-dir tokenizer --w4-dir w4_base --out-dir wt2_test_replay
python scripts/audit_recall_native_prefill_wt2_test_v1.py \
  --output-dir wt2_test_replay
```

W4 uses the bundled w4_base and tokenizer directories. It needs no
original source checkpoint or repeated quantization.

For a GitHub checkout, use --release-root . and existing source/package
paths instead of test_runtime. Each GPU replay requires a fresh output directory.

Raw token IDs are not redistributed in this docs overlay. To independently
audit the recorded reports, copy them to a fresh directory and omit only the
previous CPU audit, then regenerate exact test tokens on CPU:

```bash
cp -R reports/wt2_test_v1 wt2_test_saved_audit
rm wt2_test_saved_audit/cpu_audit_v1.json
CUDA_VISIBLE_DEVICES= python scripts/prepare_recall_wt2_test_inputs_v1.py tokens \
  --model-kind w4 --release-root test_runtime \
  --tokenizer-source-dir tokenizer --out-dir wt2_test_saved_audit
python scripts/audit_recall_native_prefill_wt2_test_v1.py \
  --output-dir wt2_test_saved_audit
```

Original checkpoint/release-tag files retain their identities. Existing
release checksum lists describe the original tagged bundle; the additional
small-file hashes are in [overlay_checksums.json](../reports/wt2_test_v1/overlay_checksums.json).
