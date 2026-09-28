# mamb2_8B_W4A16_Recall

Independent W4A16 quantization of pure NVIDIA Mamba2-8B, followed by a Resurface-inspired recall adapter. The target includes redistributable model weights.

## Base selection

**Selected source:** [nvidia/mamba2-8b-3t-4k](https://huggingface.co/nvidia/mamba2-8b-3t-4k), under Apache-2.0, pinned at `b915550c63ba9359f88f44d1f6a600d85af27302`.

The checkpoint contains 8,236,999,680 parameters: 56 pure Mamba2 blocks, width 4096, eight SSM groups, and separate 256K embedding/output matrices. Our source-native runtime uses FP16 weights and activations.

We searched for an existing public W4A16 checkpoint with permissive redistribution terms. This search found none meeting all requirements. The public Quamba2 W4A16 checkpoint was downloaded and inspected, then **rejected** because its attached UT Austin Research License does not satisfy the requested redistribution scope. Its code and weights are not used for this project. See [search record](docs/PUBLIC_BASE_SEARCH.md).

## Independent W4 format

Quantize all 114 large matrices (embedding, output head, and two projections per block) into packed affine INT4 codes, group size 128. Store FP16 scale and offset per group: approximately **4.25 bits per quantized weight**, plus headers. The 393 small tensors stay FP16.

Group parameters minimize weight-space reconstruction error across a fixed clipping grid. Quantization uses no evaluation text. We first evaluate the actual packed-file decoding using a native FP16 runtime. **This reference expands weights in GPU memory; packed storage size is not GPU residency.**

Measured base package: **4,381,415,300 bytes** including manifest (4.381GB, about 3.76× smaller than the FP16 parameter payload). All 507 tensor files passed byte hashes and exact GPU-write/CPU-read FP16 reconstruction hashes. See [manifest](reports/w4_base_v1_manifest.json).

## Resurface and verification

The frozen W4 base receives a post-D, pre-gated-RMSNorm, memoryless head-mixing adapter at all 56 layers. The adapter has 1,154,104 parameters. Training uses 1,536 successful updates and a frozen copy of the same W4 base as prose teacher. The final checkpoint is the sole candidate.

Quality evaluation pairs the W4 base and its serialized FP16 adapter on 130 WikiText-2 validation windows (264,764 targets) and 768 numeric multi-key recall prompts (384 normal, 384 target-removed). This is a previously observed benchmark family, not an untouched generalization test.

**Status: packed model generation and discarded GPU training smoke have passed; formal adapter training is running. No full W4 PPL/MK result or released adapter is available yet.** Previous FP16 or E8/W5 results must not be substituted for this model's measurements.

- [Reproduction commands](docs/REPRODUCTION.md)
- [Frozen protocol](docs/PROTOCOL.md)
- [Checklist](PLAN.md)

## Licenses

The original NVIDIA model is Apache-2.0; retain its required notices and identify modifications when distributing derived quantized weights. Project adapter/framework code follows the inherited GPL-3.0 license in [LICENSE](LICENSE). Quamba materials are excluded from the implementation and distribution.
