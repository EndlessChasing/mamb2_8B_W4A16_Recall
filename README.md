# mamb2_8B_W4A16_Recall

## Official WikiText-2 test PPL

| Fixed published model | PPL ↓ |
| --- | ---: |
| Without Resurface | 7.90768943 |
| With published Resurface | **7.51706425** |

**Official test split · 147 reset windows · 300,963 next-token targets.**
Original native SSD parallel prefill; no state rounding after each token.
[Test results and reproduction](docs/WT2_TEST_V1_RESULTS.md) ·
[Paired raw evaluation](reports/wt2_test_v1/comparison.json) ·
[CPU audit](reports/wt2_test_v1/cpu_audit_v1.json).
Historical validation PPL and synthetic CONFIRM MK results are retained below.




Independent W4A16 quantization of pure NVIDIA Mamba2-8B, followed by a Resurface-inspired recall adapter. The quantized weights and independently trained adapter tensors use Apache-2.0.

## Base selection

**Selected source:** [nvidia/mamba2-8b-3t-4k](https://huggingface.co/nvidia/mamba2-8b-3t-4k), under Apache-2.0, pinned at `b915550c63ba9359f88f44d1f6a600d85af27302`.

The checkpoint contains 8,236,999,680 parameters: 56 pure Mamba2 blocks, width 4096, eight SSM groups, and separate 256K embedding/output matrices. Our source-native runtime uses FP16 weights and activations.

We searched for an existing public W4A16 checkpoint with permissive redistribution terms. This search found none meeting all requirements. The public Quamba2 W4A16 checkpoint was downloaded and inspected, then **rejected** because its attached UT Austin Research License does not satisfy the requested redistribution scope. Its code and weights are not used for this project. See [search record](docs/PUBLIC_BASE_SEARCH.md).

## Independent W4 format

Quantize all 114 large matrices (embedding, output head, and two projections per block) into packed affine INT4 codes, group size 128. Store FP16 scale and offset per group: approximately **4.25 bits per quantized weight**, plus headers. The 393 small tensors stay FP16.

Group parameters minimize weight-space reconstruction error across a fixed clipping grid. Quantization uses no evaluation text. We first evaluate the actual packed-file decoding using a native FP16 runtime. **This reference expands weights in GPU memory; packed storage size is not GPU residency.**

Measured base package: **4,381,415,300 bytes** including manifest (4.381GB, about 3.76× smaller than the FP16 parameter payload). All 507 tensor files passed byte hashes and exact GPU-write/CPU-read FP16 reconstruction hashes. See [manifest](reports/w4_base_v1_manifest.json).

## Resurface and verification

The frozen W4 base receives a post-D, pre-gated-RMSNorm, memoryless head-mixing adapter at all 56 layers. The adapter has 1,154,104 parameters. Training completed 1,536 successful updates (1,542 attempts, six overflow retries), using a frozen copy of the same W4 base as prose teacher. The final checkpoint is the sole candidate. Its 224 serialized FP16 tensors occupy a 2,374,271-byte file. This independently implemented post-D variant is inspired by the [Resurface reference](https://github.com/Oso1106/Resurface-Multi-Binding-Recall-Is-Latent-in-Mamba-s-State).

Quality evaluation pairs the W4 base and its serialized FP16 adapter on 130 WikiText-2 validation windows (264,764 targets) and 768 numeric multi-key recall prompts (384 normal, 384 target-removed). This is a previously observed benchmark family, not an untouched generalization test.

**Status: quantization, training, full paired evaluation and independent report audit are complete.**

| Model | WikiText-2 validation PPL ↓ | Synthetic CONFIRM MK ↑ |
|---|---:|---:|
| Source FP16, historical | 7.33418 | 147/384 (38.28%) |
| Source FP16 + Resurface, historical | 7.05206 | 365/384 (95.05%) |
| Independent W4A16 | 8.01210 | 141/384 (36.72%) |
| **Independent W4A16 + Resurface** | **7.61405** | **361/384 (94.01%)** |

On validation, the W4 adapter reduces PPL by **4.97%** and improves recall by **57.29 percentage points** against its own base. Its validation PPL remains **7.97% higher** than the historical FP16 + Resurface control. All four arms score 0/384 on target-removed controls. The historical comparison passes all recorded prompt/window identity checks; its GPU execution was not rerun for this experiment.

The independent semantic audit passes **75,146 checks**, including raw metric recomputation, all packed file hashes, serialized adapter identity and training schedule. A fresh-process inference CLI replay reproduces one evaluation case exactly. See [results and scope](docs/RESULTS.md), [full W4 report](reports/w4_resurface_v1_confirm_full_reference.json) and [audit receipt](reports/semantic_audit_v1.json).

- [Reproduction commands](docs/REPRODUCTION.md)
- [Frozen protocol](docs/PROTOCOL.md)
- [Checklist](PLAN.md)

## Artifacts and runtime

- The trained [adapter](pretrained/w4_resurface_v1/) is included in this repository with its license and exact base binding.
- Download the complete packed base, matching adapter, tokenizer and runtime from [Hugging Face](https://huggingface.co/EndlessChasing/Mamb2_8B_W4A16_Recall/tree/v0.1.0) or the [GitHub Release](https://github.com/EndlessChasing/mamb2_8B_W4A16_Recall/releases/tag/v0.1.0). The large weights are release assets rather than Git history. See [download and inference commands](docs/HF_MODEL_CARD.md#download-and-generate).
- Base plus adapter: **4,383,789,571 bytes**, excluding the tokenizer and outer license/docs.
- The reference decodes the weights to FP16. Paired evaluation peak allocated GPU memory was **17,076,597,760 bytes** (about 17.08 GB); training peak was **34,596,873,216 bytes** (about 34.60 GB). These measurements do not describe a packed-resident INT4 kernel.

## Publication verification

Release **v0.1.0** is public on both platforms. Anonymous verification matched all **564 bundle files** on Hugging Face and **8 release assets** on GitHub. The complete unpacked bundle is **4,392,453,043 bytes**, including tokenizer, runtime and documentation. See [publication record](docs/PUBLICATION.md).

## Licenses

Original and independently quantized base weights, plus the trained adapter tensors: **Apache-2.0**. Retain required notices and identify modifications, as described in [weight provenance](docs/WEIGHTS_NOTICE.md). Framework, quantization and adapter implementation code: **GPL-3.0**, in [LICENSE](LICENSE). Quamba materials are excluded from the implementation and distribution.
