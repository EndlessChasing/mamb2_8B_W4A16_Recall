# Checklist

- [x] Search public pure Mamba2-8B W4A16 checkpoints and inspect licenses.
- [x] Apply user's mandatory requirement: freely redistributable weights.
- [x] Reject the Quamba candidate; select original NVIDIA Apache-2.0 source for independent quantization.
- [x] Declare independent W4 format, training and quality protocol before execution.
- [x] Implement packed W4 codec and validate independent roundtrip/math tests.
- [x] Produce packed model with full source/file hashes and measured byte counts.
- [x] Verify frozen-base identity, zero-adapter equivalence and training gradients.
- [x] Train Resurface for 1,536 successful updates.
- [x] Evaluate actual serialized adapter against W4 base on full PPL and MK.
- [x] Report actual quality, storage, GPU memory and limitations.
- [x] Publish the complete base, adapter, tokenizer, code and evidence on Hugging Face and GitHub Release v0.1.0; verify public access and all remote file identities.
