# W4 Resurface adapter v1

This directory contains the final serialized FP16 adapter trained for the
independently quantized pure NVIDIA Mamba2-8B base. It contains adapter tensors;
the quantized base and tokenizer must be obtained separately using the
[reproduction commands](../../docs/REPRODUCTION.md#use-the-included-pretrained-adapter).

| Item | Identity |
|---|---|
| Adapter file | `adapter_fp16.pt` |
| Adapter bytes | 2,374,271 |
| Adapter SHA256 | `6d38c24a102cbf83c5f1dfe4362a0b5c5e779e793a27bccd4e592ff6c9de7ca1` |
| Required W4 manifest SHA256 | `3add3f79f19d2da181c700680500390f773a47b2785d8f6e0ccaaf2ddd7bbc05` |
| Tokenizer SHA256 | `5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09` |
| Trainable parameters | 1,154,104, exported in 224 FP16 tensors |
| Training | 1,536 successful updates; final checkpoint only |
| Adapter behavior | Memoryless post-D head mixing, soft sigmoid gate, all 56 layers |

The adapter file is copied byte for byte from the final verified training
export. Use the matching W4 base and tokenizer. An adapter for the original
FP16 model or another quantizer is not a substitute. Regenerate the base with
the pinned code, require the manifest SHA256 above, and stop on any mismatch;
do not edit bindings or replace a generated manifest to force compatibility.

The implementation is inspired by
[Resurface](https://github.com/Oso1106/Resurface-Multi-Binding-Recall-Is-Latent-in-Mamba-s-State)
and uses a post-D insertion site. This is an independently implemented variant.
For quality measurements and their scope, use the repository's completed
paired evaluation reports; training completion alone is not a quality result.

The adapter tensors are **Apache-2.0**, with the license and provenance in
[LICENSE.txt](LICENSE.txt) and [NOTICE.md](NOTICE.md). Repository implementation
code remains **GPL-3.0**. The numerical reference expands the W4 base into FP16
GPU weights; this adapter does not add a packed-resident W4 kernel.
