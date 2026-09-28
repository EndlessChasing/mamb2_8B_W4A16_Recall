# Independent W4A16 Mamba2-8B Resurface protocol

Declared 2026-09-28 before quantization, training or quality evaluation of this repository's independent W4 candidate. The user requires redistributable weights. No Quamba weights, rotation, implementation or adapter are dependencies.

## Frozen model and quantization

Source: NVIDIA `nvidia/mamba2-8b-3t-4k`, Apache-2.0, revision
`b915550c63ba9359f88f44d1f6a600d85af27302`, checkpoint SHA256
`47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`;
tokenizer SHA256 `5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09`.
This is the pure 56-layer Mamba2-8B model, with untied 256K embedding/head.

Independently quantize all 114 large matrices (56 input projections, 56 output
projections, embedding and head) to affine uniform four-bit codes. Groups contain
128 adjacent input-axis weights. Each group stores a FP16 scale and FP16 offset;
two codes are packed per byte. Metadata adds 0.25 bits per quantized weight,
so these matrices require approximately 4.25 bits/weight before file headers.
The other 393 small tensors remain FP16. Do not describe every parameter as INT4.

For each group, fit using weight-space MSE only, over fixed range factors
1.00, .99, .98, .97, .96, .95, .94, .92, .90 around its min/max midpoint.
Compare actual serialized FP16 scale/offset and FP16 reconstruction. No prose,
MK data, validation labels or source activation data enter quantization.
This fixed quantizer defines one candidate; no evaluation-based quantizer search.
Constant/zero groups, nonfinite inputs and truncated files must be handled explicitly.
Write/read actual packed files, verify exact reconstruction and per-file hashes,
and bind the resulting manifest to both training and evaluation.

Quality execution decodes the saved W4 package into frozen FP16 native weights.
This is a W4A16 numerical reference, not a packed-resident GPU kernel. Native
FP16 residual orchestration and FP16 state cache match our existing source control.
The source BF16-to-FP16 runtime is not asserted to reproduce Megatron BF16 exactly.
No rotation, residual weight correction or small-tensor readaptation is included.

Use the same post-D adapter as the compressed experiment at all 56 native
gated RMSNorm inputs: `y + sigmoid(w dot u+b) * g*(V@y)` across 128 heads.
Its 224 tensors contain 1,154,104 FP32 trainable master parameters, exported
as FP16 for inference. Initialize V=0, g=1, w=0, b=-4; use soft sigmoid at
train and eval, no task switch, EMA, or new recurrent cache. The adapter is
external to the frozen base; check its identities and absence of base grads.

## Training

Reuse the *same exact data generator contract and seeds* as the compressed
adapter: 1,536 TRAIN numeric bindings over three templates and N=16/64,
disjoint six-digit key/value intervals; full 256K CE on every answer suffix
token. Use seed 2026092803 and `torch.randperm(1536)` once for example order.
Each step pairs one MK example with a 512-token segment from the already
prepared 448 WikiText-2 TRAIN windows. Their historical manifest and file
SHA256 are pinned by the train command. Heldout and validation windows are
not included in training. Reusing this fixed TRAIN text gives the control the
same training exposures as the compressed arm.

Use a second, independently loaded **unadapted W4-decoded FP16** model as the frozen
prose teacher. This is the same own-unadapted-base teacher policy as our previous experiments. The fixed per-step loss is
`MK answer CE + 0.5 prose CE + 0.5 KL(own unadapted teacher || student) + 3 C`,
where C is the same prose-only router closure, with budget 0.006 and excess
coefficient 10. Temperature 1, 511 prose targets per step, 64-token staged
head chunks, and one optimizer update after both task gradients.

AdamW FP32 masters: V/g LR 1e-4, w/b LR 3e-4, betas (.9,.999), eps 1e-8,
weight decay 0, clip norm 1. Multiplicative schedule
`0.1+0.9*(1+cos(pi*j/1535))/2` at successful index j. Native FP16 forward,
block checkpointing, gradient scale 1024 with growth interval 2000. Overflow
retries the exact same pair without optimizer/master update, at most 8 retries
and 1,544 total attempts. Run 1,536 **successful** updates; the final step is
the only candidate. Small GPU smokes are discarded before formal training.

## Evaluation and interpretation

After training, restore the actual serialized FP16 adapter onto the same
W4-decoded base. Evaluate baseline, adapter enabled and restored baseline
in a single process, with identical tokenizer, MK prompts and full WikiText-2
validation windows. For MK use full 256K greedy generation up to 12 tokens,
first standalone six-digit number, normal and target-removed controls, native
prefill plus recurrent decode with fresh FP16 cache. For PPL use all 130
nonoverlapping reset windows and all 264,764 next-token targets. Record all
raw scores, differences and file hashes; no early selection or task switch.

The earlier compressed arm's independent CONFIRM set has now been observed and
uses the same three template families. Running this W4 experiment on it gives
a matched historical comparison, **not a newly untouched holdout**. The
WikiText-2 validation text also informed earlier development. We will not
promote a W4+adapter result to a general recall conclusion without new
templates, longer distances and a truly untouched corpus.

Report the new W4 base and W4+adapter together, with matched source FP16 historical metrics clearly identified as historical. Verify prompt/window hashes before comparing them. Report PPL and recall together; no packed-runtime speed or memory claim follows from these quality measurements.
