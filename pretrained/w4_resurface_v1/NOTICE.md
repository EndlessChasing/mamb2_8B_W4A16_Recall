# Adapter provenance and license notice

Copyright 2026 EndlessChasing.

The independently trained adapter tensors in `adapter_fp16.pt` are licensed
under the Apache License, Version 2.0, reproduced in [LICENSE.txt](LICENSE.txt).
This grant applies to the weight artifact identified by SHA256
`6d38c24a102cbf83c5f1dfe4362a0b5c5e779e793a27bccd4e592ff6c9de7ca1`.
The repository's framework and adapter implementation code remain GPL-3.0.

The adapter was trained on the project's independently quantized W4 base of
NVIDIA `nvidia/mamba2-8b-3t-4k`, revision
`b915550c63ba9359f88f44d1f6a600d85af27302`. The original model is Apache-2.0
according to its
[pinned model card](https://huggingface.co/nvidia/mamba2-8b-3t-4k/blob/b915550c63ba9359f88f44d1f6a600d85af27302/README.md).
The required W4 base manifest SHA256 is
`3add3f79f19d2da181c700680500390f773a47b2785d8f6e0ccaaf2ddd7bbc05`.
The base modification quantizes 114 large matrices to affine group-128 INT4
with FP16 scale and offset, while retaining 393 small tensors in FP16.

Original model reference: *An Empirical Study of Mamba-based Language Models*,
Waleffe et al., 2024, [arXiv 2406.07887](https://arxiv.org/abs/2406.07887).
Adapter design inspiration:
[Resurface](https://github.com/Oso1106/Resurface-Multi-Binding-Recall-Is-Latent-in-Mamba-s-State).
The project uses an independently implemented post-D variant, with memoryless
head mixing at all 56 layers, trained specifically for this quantized base.
The upstream authors do not endorse this modified model. No Quamba software
or weights are included.

Retain this notice and the Apache-2.0 license when redistributing the adapter.
For a complete model bundle, also retain the base's exact manifest, original
attributions and modification notice described in
[WEIGHTS_NOTICE.md](../../docs/WEIGHTS_NOTICE.md). Keep outer license documents
separate from the bound core W4 package, whose hash must remain unchanged.
