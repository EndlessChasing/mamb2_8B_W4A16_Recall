# mamb2_8B_W4A16_Recall

Research project: apply a Resurface-inspired recall adapter to an existing public W4A16 quantization of pure Mamba2-8B.

## Public base

[ut-enyac/quamba2-8b-converted-w4a16](https://huggingface.co/ut-enyac/quamba2-8b-converted-w4a16), from the [official Quamba project](https://github.com/enyac-group/Quamba).

- Pinned revision: `997f760f29c10574ee8363b8680d636048dee048`.
- Source architecture: NVIDIA `mamba2-8b-3t-4k`; 56 Mamba2 layers, width 4096, eight SSM groups, no attention layers, untied 256K embedding and output head.
- Published configuration: `W4A16QMamba2`, `W4O16Embedding`, `W4A16B16O16Linear`.
- Weight file: **4,253,601,410 bytes** (4.254 GB / 3.962 GiB). This is file size, not measured runtime GPU memory.
- Weight SHA256: `0e0097aeffe8da48a21bf9698e32f474f54f766b902661b5626b366a8d5c4bc4`.
- The full weight download has been hashed and its 734 tensor entries inspected. See [receipt](reports/public_base_receipt.json).

```bash
hf download ut-enyac/quamba2-8b-converted-w4a16 \
  --revision 997f760f29c10574ee8363b8680d636048dee048 \
  --local-dir models/quamba2-w4a16
```

## License and distribution

The upstream checkpoint includes the [UT Austin Research License](https://huggingface.co/ut-enyac/quamba2-8b-converted-w4a16/blob/997f760f29c10574ee8363b8680d636048dee048/license.txt). Public download availability must not be interpreted as permission for unrestricted redistribution. Its terms cover research/personal use and restrict redistribution of the software and derivative products. We have not established permission to republish the checkpoint or adaptations.

This repository currently contains project documentation and factual metadata only. It does not redistribute upstream weights or Quamba source code. Licensing and the intended distribution scope must be resolved before publishing an adaptation.

## Status

**No W4A16 Resurface adapter has been trained or validated yet.** Metrics from our separate FP16 and E8/W5 projects do not establish quality for this base.

The integration must preserve Quamba's rotations, online Hadamard transform and FP32 residual accumulation. Its inference-only quantized operators need a verified differentiable bridge for adapter training. A dense FP16 decoding reference would not demonstrate packed W4 GPU residency or Quamba kernel performance.

See [PLAN.md](PLAN.md) for remaining work.
