# v0.1.0 — complete W4A16 Mamba2-8B + recall adapter

This release packages the complete independently quantized pure NVIDIA Mamba2-8B base, its matching Resurface-inspired adapter, the original tokenizer, custom loading code and evaluation evidence. Weight artifacts are **Apache-2.0**; implementation code is **GPL-3.0**.

## Results

| Model | WikiText-2 PPL ↓ | Numeric MK recall ↑ |
|---|---:|---:|
| Source FP16, historical | 7.33418 | 147/384 (38.28%) |
| Source FP16 + Resurface, historical | 7.05206 | 365/384 (95.05%) |
| Independent W4A16 | 8.01210 | 141/384 (36.72%) |
| **Independent W4A16 + Resurface** | **7.61405** | **361/384 (94.01%)** |

PPL includes all 130 WikiText-2 validation reset windows and 264,764 targets. MK includes 384 normal and 384 target-removed prompts; all four arms score 0/384 on the removed controls. The CONFIRM instances and template family were previously observed. Historical comparisons have matching recorded prompt/window identities; historical GPU execution was not rerun.

The adapter improves its own W4 baseline's recall by 57.29 percentage points and lowers PPL by 4.97%. Against the historical FP16 + Resurface control, it retains a 7.97% higher PPL and 1.04 percentage points lower recall. See [complete results](https://github.com/EndlessChasing/mamb2_8B_W4A16_Recall/blob/main/docs/RESULTS.md).

## Download the complete model

### Hugging Face

```bash
hf download EndlessChasing/Mamb2_8B_W4A16_Recall \
  --revision v0.1.0 --local-dir Mamb2_8B_W4A16_Recall
cd Mamb2_8B_W4A16_Recall
```

[Hugging Face model and usage instructions](https://huggingface.co/EndlessChasing/Mamb2_8B_W4A16_Recall)

### GitHub release assets

Download all three archive parts and the checksum file from this release:

- `mamb2_8B_W4A16_Recall-v0.1.0.tar.part-00`
- `mamb2_8B_W4A16_Recall-v0.1.0.tar.part-01`
- `mamb2_8B_W4A16_Recall-v0.1.0.tar.part-02`
- `release-assets.sha256`

Keep the files in the same directory. On Linux, verify and extract:

```bash
sha256sum --check release-assets.sha256
cat mamb2_8B_W4A16_Recall-v0.1.0.tar.part-* | tar -xf -
cd mamb2_8B_W4A16_Recall-v0.1.0
sha256sum --check SHA256SUMS
```

On macOS, replace each `sha256sum --check FILE` with `shasum -a 256 -c FILE`. Each part is at most 1,500,000,000 bytes. The parts are consecutive pieces of one tar archive; extract them together in numerical order. Keep all model files and the manifest unchanged.

## Generate

Run on CUDA Linux with CUDA-compatible PyTorch and working native `mamba-ssm` extensions. The verified environment is recorded in `reports/environment_v1.json`; native Mamba installation is separate from the project package.

From the downloaded or extracted bundle root:

```bash
python -m pip install -e code
python code/scripts/generate.py \
  --w4-dir w4_base \
  --tokenizer tokenizer/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
  --adapter adapter/adapter_fp16.pt \
  --prompt 'The key idea of a state space model is' \
  --max-new-tokens 64
```

Omit `--adapter` for the W4 baseline. This is a base completion model with a custom loader. The loader checks packed files and exact adapter/base/tokenizer compatibility. The original full-precision checkpoint is unnecessary for inference.

## Storage, runtime and verification

- **Core W4 base:** 4,381,415,300 bytes, including its manifest and 507 tensor files.
- **Adapter:** 2,374,271 bytes, containing 1,154,104 parameters in 224 FP16 tensors.
- **Base + adapter:** 4,383,789,571 bytes (4.384 GB). Tokenizer, code, documentation and archive overhead are additional.
- **Quantization:** 114 large matrices use packed affine INT4 with 128-weight groups and FP16 scale/offset. These matrices use approximately 4.25 bits per weight before headers; 393 small tensors stay FP16.
- **Runtime:** the reference expands weights into FP16. Paired evaluation peak allocated GPU memory was 17,076,597,760 bytes (17.08 GB). There is no packed-resident INT4 inference kernel or measured speedup claim.
- **Verification:** all packed files passed byte hashes and decoded reconstruction hashes; the independent semantic audit passed 75,146 checks. A fresh-process CLI replay reproduced one recorded evaluation case exactly.

The full bundle includes `SHA256SUMS` and `RELEASE_MANIFEST.json`. Exact model identities:

```text
W4 manifest SHA256:
3add3f79f19d2da181c700680500390f773a47b2785d8f6e0ccaaf2ddd7bbc05
Adapter SHA256:
6d38c24a102cbf83c5f1dfe4362a0b5c5e779e793a27bccd4e592ff6c9de7ca1
Tokenizer SHA256:
5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09
```

## Provenance and limits

Source: [NVIDIA mamba2-8b-3t-4k](https://huggingface.co/nvidia/mamba2-8b-3t-4k/tree/b915550c63ba9359f88f44d1f6a600d85af27302), revision `b915550c63ba9359f88f44d1f6a600d85af27302`, with 8,236,999,680 parameters and 56 pure Mamba2 layers. The source BF16 weights are cast to FP16 for this project's quantization and numerical reference.

The adapter is an independently implemented post-D variant inspired by [Resurface](https://github.com/Oso1106/Resurface-Multi-Binding-Recall-Is-Latent-in-Mamba-s-State), trained specifically for this W4 base. Quamba software and checkpoints are excluded.

Weights and tokenizer retain Apache-2.0 provenance; project weight contributions are Apache-2.0. Bundled implementation code remains GPL-3.0. See the bundle's `LICENSE`, `NOTICE` and `code/LICENSE`.

These are matched, previously observed benchmark results. New task families, longer-distance recall, broad downstream quality and alternate runtime/hardware behavior remain unevaluated. The observed results do not establish equivalence to unquantized quality or a 4.384 GB GPU memory requirement.
