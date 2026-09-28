# Checklist

- [x] Find an existing publicly downloadable pure Mamba2-8B W4A16 checkpoint.
- [x] Pin revision, confirm architecture, inspect artifact inventory and license.
- [x] Download the complete published checkpoint and verify its SHA256.
- [ ] Resolve research-only versus freely redistributable base requirement.
- [ ] Validate loading, packing, rotations, numerical precision and training gradients.
- [ ] Freeze adapter training/evaluation protocol with all source hashes.
- [ ] Train Resurface with the base weights frozen.
- [ ] Compare W4A16 baseline and serialized adapter on full WikiText-2 PPL and paired numeric MK.
- [ ] Report measured quality, file size, GPU memory and validation limitations together.
- [ ] Publish only artifacts whose redistribution terms have been established.
