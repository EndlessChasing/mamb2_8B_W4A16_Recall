# W4A16 Resurface result audit

Integrity audit: **PASS** (75146 checks, 0 errors).

| Arm | PPL | Normal recall | Target removed |
|---|---:|---:|---:|
| W4 base | 8.012096 | 141/384 (36.72%) | 0/384 |
| W4 + Resurface | 7.614047 | 361/384 (94.01%) | 0/384 |
| Historical source FP16 base | 7.334176 | 147/384 (38.28%) | 0/384 |
| Historical source FP16 + Resurface | 7.052064 | 365/384 (95.05%) | 0/384 |

Full WikiText-2 validation: 130 reset windows, 264,764 next-token targets. MK: 384 normal plus 384 target-removed cases; previously observed template family and instances.

## Scope

- CPU semantic audit of supplied reports and actual serialized adapter; no GPU replay.
- PPL is recomputed from raw window NLL and target counts; recall from recorded generated text.
- Training schedule and reported frozen-base status are checked; optimizer execution and all base bytes are not independently replayed.
- Previously observed CONFIRM instances and WikiText validation; no untouched-generalization claim.
- Packed W4 file size is separate from the decoded FP16 reference runtime memory.

## Recall by binding count

| Arm | N=16 | N=64 |
|---|---:|---:|
| W4 base | 110/192 | 31/192 |
| W4 + Resurface | 192/192 | 169/192 |
| Historical source FP16 base | 113/192 | 34/192 |
| Historical source FP16 + Resurface | 191/192 | 174/192 |

The W4 adapter gains 222 answers and loses 2 relative to its own base, for a
net gain of 220/384. PPL decreases 4.9681%. Compared with the matched historical
FP16 + Resurface control, W4 + Resurface PPL is 7.9691% higher and recall is
1.0417 percentage points lower. No equivalence to unquantized quality is claimed.

## Storage and measured GPU memory

| Item | Bytes |
|---|---:|
| Packed tensor files (507) | 4,381,242,635 |
| Bound manifest | 172,665 |
| Base core package | 4,381,415,300 |
| Serialized FP16 adapter | 2,374,271 |
| Base core plus adapter | 4,383,789,571 |
| Expanded FP16 base parameter payload | 16,473,999,360 |
| Paired evaluation peak allocated GPU memory | 17,076,597,760 |
| Training peak allocated GPU memory | 34,596,873,216 |
| Native FP16 SSM + convolution cache, batch 1 | 122,028,032 |

Outer license documents, tokenizer and other documentation are excluded from
weight-package totals. The format averages 4.25518 bits per original model
parameter including tensor-file headers, before manifest and adapter. The
current runtime expands weights to FP16; a packed-resident INT4 kernel is not
implemented. PPL uses parallel native SSD prefill, while MK uses native prefill
followed by recurrent token decoding with a fresh FP16 cache.

## Evidence and artifact identities

- [Full paired W4 evaluation](../reports/w4_resurface_v1_confirm_full_reference.json)
- [Semantic audit](../reports/semantic_audit_v1.json)
- [Training report](../reports/w4_resurface_v1_train.json)
- [Historical FP16 paired evaluation](../reports/historical_source_confirm_full_reference.json)
- [Completed execution stages and code hashes](../reports/experiment_v1_reference.json)
- [Fresh-process CLI replay](../reports/generation_cli_replay_v1.json)
- [Mac packed-file backup verification](../reports/w4_local_backup_receipt.json)
- [Apache-2.0 pretrained adapter](../pretrained/w4_resurface_v1/)

W4 manifest SHA256:
`3add3f79f19d2da181c700680500390f773a47b2785d8f6e0ccaaf2ddd7bbc05`.
Adapter SHA256:
`6d38c24a102cbf83c5f1dfe4362a0b5c5e779e793a27bccd4e592ff6c9de7ca1`.
Raw W4 evaluation SHA256:
`f49f205ab684f883a83d3122b1c84904c71c09ac4c5ed197b137b5c5c423da41`.

The trained adapter and evaluation reports are distributed in this repository.
Download the complete packed base, adapter, tokenizer and runtime from
[Hugging Face](https://huggingface.co/EndlessChasing/Mamb2_8B_W4A16_Recall/tree/v0.1.0)
or the [GitHub Release](https://github.com/EndlessChasing/mamb2_8B_W4A16_Recall/releases/tag/v0.1.0).
See [download/inference instructions](HF_MODEL_CARD.md), [reproduction](REPRODUCTION.md)
and [license scope](WEIGHTS_NOTICE.md).
