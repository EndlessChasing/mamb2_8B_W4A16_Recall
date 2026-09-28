---
license: apache-2.0
base_model:
- nvidia/mamba2-8b-3t-4k
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
tags:
- mamba
- mamba2
- quantized
- w4a16
- resurface
- recall
---

# Mamb2_8B_W4A16_Recall

A complete, independently quantized **pure Mamba2-8B** checkpoint with a small Resurface-inspired recall adapter. This bundle contains the packed base, matching adapter, tokenizer, loading code and evaluation evidence. Inference does not require downloading the original 16.47 GB checkpoint or repeating quantization.

- **Base + adapter weight storage:** 4,383,789,571 bytes (4.384 GB), excluding tokenizer, code and documentation.
- **Measured result:** WikiText-2 PPL **7.61405**; numeric multi-key recall **361/384 (94.01%)**.
- **Runtime:** the supplied CUDA reference expands weights into FP16. Its measured paired-evaluation peak allocated GPU memory is **17.08 GB**. A packed-resident INT4 inference kernel is not included.
- **Licenses:** Apache-2.0 for the weight artifacts; GPL-3.0 for the implementation code in `code/`.

[GitHub source](https://github.com/EndlessChasing/mamb2_8B_W4A16_Recall) · [Results](docs/RESULTS.md) · [Frozen experiment protocol](docs/PROTOCOL.md)

## Download and generate

Use a CUDA Linux environment with working native [Mamba CUDA extensions](https://github.com/state-spaces/mamba#installation). The verified environment used Python 3.10.12, PyTorch 2.11.0+cu128, mamba-ssm 2.3.2.post1 and an RTX PRO 6000 Blackwell Server Edition. See [environment receipt](reports/environment_v1.json). Installation and bitwise behavior on other environments have not been validated.

```bash
hf download EndlessChasing/Mamb2_8B_W4A16_Recall \
  --revision v0.1.0 --local-dir Mamb2_8B_W4A16_Recall
cd Mamb2_8B_W4A16_Recall

# Install a CUDA-compatible PyTorch and native mamba-ssm first.
python -m pip install -e code
python code/scripts/generate.py \
  --w4-dir w4_base \
  --tokenizer tokenizer/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
  --adapter adapter/adapter_fp16.pt \
  --prompt 'The key idea of a state space model is' \
  --max-new-tokens 64
```

Omit `--adapter` to run the quantized baseline. The command emits JSON with the completion and generated token IDs. This is a base completion model. It has no instruction tuning or chat template. The helper accepts a maximum of 4,096 tokens for prompt plus completion; this limit is not a claim of validated 4K recall quality.

The loader checks all packed tensor files and the adapter's base/tokenizer bindings. Keep the `w4_base/` manifest and files unchanged. `SHA256SUMS` provides additional bundle file checksums. On Linux use `sha256sum --check SHA256SUMS`; on macOS use `shasum -a 256 -c SHA256SUMS`. The generation runtime itself was validated on CUDA Linux.

## Model and quantization

The source is NVIDIA's Apache-2.0 [mamba2-8b-3t-4k](https://huggingface.co/nvidia/mamba2-8b-3t-4k/tree/b915550c63ba9359f88f44d1f6a600d85af27302), pinned at revision `b915550c63ba9359f88f44d1f6a600d85af27302`. It contains **8,236,999,680 parameters**, 56 pure Mamba2 blocks, hidden width 4,096, eight SSM groups and separate 256K embedding/output matrices.

We quantize 114 large matrices—56 input projections, 56 output projections, embedding and output head—to unsigned affine INT4 codes. Groups of 128 adjacent input-axis weights each retain an FP16 scale and FP16 offset. Two codes occupy one byte. This is approximately **4.25 bits per quantized weight** before headers. The remaining 393 small tensors retain FP16 precision.

A fixed clipping grid minimizes weight reconstruction MSE using the actually serialized FP16 scale, offset and reconstruction. Quantization uses no calibration text or evaluation data. The numerical reference first casts the original BF16 weights to FP16. Its results are not an asserted reproduction of the original Megatron BF16 runtime.

The codec and weights were produced independently. Quamba software and quantized checkpoints are not dependencies or distributed artifacts.

## Recall adapter

The frozen quantized model receives memoryless mixing across 128 heads at all 56 layers, after the D skip contribution and before gated RMSNorm. This independently implemented post-D variant is inspired by [Resurface](https://github.com/Oso1106/Resurface-Multi-Binding-Recall-Is-Latent-in-Mamba-s-State). It is not an exact reproduction of the reference insertion site.

The adapter has **1,154,104 parameters**, exported as 224 FP16 tensors. It uses a soft sigmoid gate during training and inference, with no task switch or extra recurrent cache. Training completed 1,536 successful updates over 1,542 attempts, including six overflow retries. The final checkpoint was the sole evaluation candidate.

Training pairs synthetic numeric binding tasks with WikiText-2 TRAIN prose. Its frozen prose teacher is a separately loaded copy of the same unadapted W4 base. See the [training report](reports/w4_resurface_v1_train.json) and [protocol](docs/PROTOCOL.md). This adapter is bound to this exact W4 base; an adapter for the original FP16 model or another quantizer is not interchangeable.

## Evaluation

| Model | WikiText-2 PPL ↓ | Normal MK recall ↑ | Target-removed controls |
|---|---:|---:|---:|
| Source FP16, historical | 7.33418 | 147/384 (38.28%) | 0/384 |
| Source FP16 + Resurface, historical | 7.05206 | 365/384 (95.05%) | 0/384 |
| Independent W4A16 | 8.01210 | 141/384 (36.72%) | 0/384 |
| **Independent W4A16 + Resurface** | **7.61405** | **361/384 (94.01%)** | **0/384** |

PPL covers **130 nonoverlapping reset windows and 264,764 next-token targets** from WikiText-2 validation. Numeric multi-key recall (MK) uses 384 normal prompts plus 384 target-removed controls, full-vocabulary greedy decoding up to 12 tokens and the first standalone six-digit answer. Normal cases include 192 prompts with 16 bindings and 192 with 64 bindings. The adapted W4 model scores **192/192** and **169/192**, respectively. Each case starts with a fresh native FP16 state cache.

Against its own W4 baseline, the adapter lowers PPL by **4.97%** and raises recall by **57.29 percentage points**. Relative to the matched historical FP16 + Resurface control, its PPL is **7.97% higher** and recall is **1.04 percentage points lower**. The historical controls pass the recorded prompt/window identity checks, but their GPU evaluations were not rerun for this experiment.

An independent semantic audit passes 75,146 checks: raw metric recomputation, training schedule checks, all packed file hashes and the serialized adapter identity. A fresh-process inference CLI replay reproduces one evaluation case exactly. See the [full paired report](reports/w4_resurface_v1_confirm_full_reference.json), [audit](reports/semantic_audit_v1.json) and [CLI replay receipt](reports/generation_cli_replay_v1.json).

### Interpretation limits

- The CONFIRM instances and template family were previously observed. WikiText-2 validation also informed earlier project development. These are matched benchmark results, not a new untouched generalization test.
- PPL uses native parallel SSD prefill. MK uses native prefill followed by recurrent decoding. We do not claim a full PPL evaluation with state rounded after every individual token.
- The 4.384 GB weight storage does not imply 4.384 GB GPU memory. The current runtime expands weights into FP16. No packed-kernel speedup, throughput or energy claim is made.
- The measured state cache is 122,028,032 bytes at batch size one; it is FP16 and has no state quantization.
- Longer-distance recall, new task families, chat behavior and broader downstream quality remain unevaluated. Quantization quality does not equal the unquantized control.

## Bundle layout and identities

```text
w4_base/       507 packed tensor files and the bound manifest.json
adapter/       adapter_fp16.pt and weight notices
tokenizer/     the original pinned SentencePiece tokenizer
code/          custom loader, generation/training/evaluation scripts; GPL-3.0
docs/          protocol, reproduction and result documentation
reports/       recorded evaluation and integrity evidence
LICENSE        Apache-2.0 for weight artifacts
NOTICE         provenance and modification notices
SHA256SUMS     bundle file checksums
```

| Artifact | Bytes |
|---|---:|
| W4 core base, including manifest | 4,381,415,300 |
| FP16 adapter file | 2,374,271 |
| Base core + adapter | 4,383,789,571 |
| Expanded FP16 base parameter payload | 16,473,999,360 |
| Paired evaluation peak allocated GPU memory | 17,076,597,760 |

SHA-256 identities:

```text
Original checkpoint:
47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb
w4_base/manifest.json:
3add3f79f19d2da181c700680500390f773a47b2785d8f6e0ccaaf2ddd7bbc05
adapter/adapter_fp16.pt:
6d38c24a102cbf83c5f1dfe4362a0b5c5e779e793a27bccd4e592ff6c9de7ca1
Tokenizer:
5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09
Frozen protocol:
9aaba5861cf818275acc1bad00c87ccab22d2d7fc3538f5f6ef5456e67c1d472
```

The training report records the additional synthetic-data, prose-data and individual adapter tensor bindings. Full reproduction from the original source is documented in the [GitHub repository](https://github.com/EndlessChasing/mamb2_8B_W4A16_Recall/blob/main/docs/REPRODUCTION.md).

## License and attribution

Original model and tokenizer: NVIDIA, under Apache-2.0 as declared by the [pinned upstream model card](https://huggingface.co/nvidia/mamba2-8b-3t-4k/blob/b915550c63ba9359f88f44d1f6a600d85af27302/README.md). Independently quantized weights and trained adapter tensors: Apache-2.0, with the grant for project contributions made by EndlessChasing. Retain the [license](LICENSE), [notice](NOTICE) and modification attributions when redistributing.

Bundled implementation code is separately licensed under [GPL-3.0](code/LICENSE). The weight license does not change the code license. Native Mamba and other dependencies retain their own licenses. The upstream authors do not endorse this quantized model or adapter.

Source model reference: Waleffe et al., *An Empirical Study of Mamba-based Language Models* (2024), [arXiv:2406.07887](https://arxiv.org/abs/2406.07887). Recall method inspiration: [Resurface reference repository](https://github.com/Oso1106/Resurface-Multi-Binding-Recall-Is-Latent-in-Mamba-s-State).
