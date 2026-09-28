# Public base search and decision

The user requires freely redistributable weights. Searches of the Hugging Face model API for `mamba`, `mamba2`, `mamba-2`, `quamba` and `mamb2`, and web searches for8B with4bit/GGUF/AWQ/GPTQ did not identify an eligible prequantized checkpoint. This is a search result, not proof that none exists.

| Candidate | Result |
|---|---|
| [ut-enyac/quamba2-8b-converted-w4a16](https://huggingface.co/ut-enyac/quamba2-8b-converted-w4a16) | Actual pure8B W4A16,4,253,601,410-byte weight file; rejected due to attached UT Austin Research License redistribution restrictions. |
| [ut-enyac/mamba2-8b-converted-uniql-1.0-masked-lora-rft-w4a16](https://huggingface.co/ut-enyac/mamba2-8b-converted-uniql-1.0-masked-lora-rft-w4a16) | Adapter, not a complete W4 base. |
| [ib-ssm/mamba2-8b-3t-4k-hf](https://huggingface.co/ib-ssm/mamba2-8b-3t-4k-hf) | Apache-2.0 converted NVIDIA model; approximately16.474GB, unquantized. |
| [devingulliver/mamba2-8b](https://huggingface.co/devingulliver/mamba2-8b) | Apache-2.0 converted NVIDIA model;16,474,164,915-byte weights, unquantized. |
| Other MLX/GGUF results | Wrong size (2.7B/2.8B/7B), or different/hybrid source architecture. |

Selected independent-quantization source: [NVIDIA official card at pinned revision](https://huggingface.co/nvidia/mamba2-8b-3t-4k/blob/b915550c63ba9359f88f44d1f6a600d85af27302/README.md), which identifies the pure8B model and Apache-2.0 license.

The Quamba download receipt and passive tensor inventory remain as factual search evidence in `reports/`. They describe a rejected candidate, not the selected base or runtime. No Quamba code or weight data are committed to this repository.
