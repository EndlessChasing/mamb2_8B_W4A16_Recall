# Checklist

- [x] Search public pure Mamba2-8B W4A16 checkpoints and inspect licenses.
- [x] Apply user's mandatory requirement: freely redistributable weights.
- [x] Reject the Quamba candidate; select original NVIDIA Apache-2.0 source for independent quantization.
- [x] Declare independent W4 format, training and quality protocol before execution.
- [ ] Implement packed W4 codec and validate independent roundtrip/math tests.
- [ ] Produce packed model with full source/file hashes and measured byte counts.
- [ ] Verify frozen-base identity, zero-adapter equivalence and training gradients.
- [ ] Train Resurface for1536 successful updates.
- [ ] Evaluate actual serialized adapter against W4 base on full PPL and MK.
- [ ] Report actual quality, storage, GPU memory and limitations.
- [ ] Publish reproducible code/results and properly licensed artifacts.
