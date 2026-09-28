# Weight provenance and distribution notice

Original model: NVIDIA `nvidia/mamba2-8b-3t-4k`.

Original model license: Apache-2.0, as declared in the [official pinned model card](https://huggingface.co/nvidia/mamba2-8b-3t-4k/blob/b915550c63ba9359f88f44d1f6a600d85af27302/README.md). A copy of the license is provided in [WEIGHTS_LICENSE.txt](WEIGHTS_LICENSE.txt).

This project modifies the original weights by independent affine INT4 quantization of114 large matrices with128-element groups, keeping393 small tensors in FP16. It adds a separately trained Resurface-inspired adapter after quantization. The original model authors do not endorse these modifications. Published metrics must identify the actual modified checkpoint and evaluation protocol.

The source model authors and paper are credited in the upstream model card: *An Empirical Study of Mamba-based Language Models*, Waleffe et al.,2024, [arXiv2406.07887](https://arxiv.org/abs/2406.07887).

When redistributing quantized base weights, include this provenance notice, the Apache-2.0 license, any applicable upstream notices and the exact source/quantization manifest. The project framework/adapter implementation is separately distributed under the repository's GPL-3.0 license. No Quamba software or quantized weights are included.
