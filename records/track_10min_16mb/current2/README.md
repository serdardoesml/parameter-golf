# Record Candidate: PR #1855 Base + Removed First MLP + MLP_MULT 4.375

**Status:** 3-seed logs pending. This folder is prepared as a submission writeup for the current `train_gpt.py` candidate.

This submission builds on the PR #1855-style stack and makes one architectural change: remove the first block's MLP and reallocate that parameter budget into the remaining MLPs by setting `MLP_MULT=4.375`.

The intuition is that after only one attention layer, the first MLP is less useful than later MLPs as there is not as much information collected from the attention circuits. The implementation removes the block-0 MLP weights entirely, not just the compute path: the MLP banks are sized for blocks 1..10, block 0 receives `None` MLP weights, and normal, parallel, TTT, unbank/rebank, and per-group serialization paths all treat block 0 as attention-only.

## Results

Fill this table from the final 3 seed logs before submission.

| Seed | Steps | Train ms | Pre-quant val_bpb | Post-quant val_bpb | Post-TTT val_bpb | Artifact bytes |
|------|------:|---------:|------------------:|-------------------:|-----------------:|---------------:|
| 42   | TODO  | TODO     | TODO              | TODO               | TODO             | TODO           |
| 0    | TODO  | TODO     | TODO              | TODO               | TODO             | TODO           |
| 1234 | TODO  | TODO     | TODO              | TODO               | TODO             | TODO           |
| **Mean** | TODO | TODO | TODO | TODO | TODO | TODO |

## What Changed

Compared with the PR #1855 base:

```text
MLP_MULT = 4.375
blocks.0.mlp removed
blocks.1..10 MLP weights retained and widened
```

Implementation details:

- `mlp_up_bank` and `mlp_down_bank` have shape `num_layers - 1`.
- `_bank_weights(0)` returns `None` for MLP up/down weights.
- `Block.forward`, `_parallel_block`, `_block_with_lora`, and `_parallel_block_with_lora` skip MLP work when `up_w is None`.
- `_unbank_state_dict` and `_rebank_state_dict` map MLP tensors over blocks `1..10`.
- Per-group compression/deserialization maps `mlp.fc.weight.q` and `mlp.proj.weight.q` over blocks `1..10`, avoiding stale `blocks.0.mlp.*` or missing `blocks.10.mlp.*` keys.

Everything else follows the PR #1855-style stack in this folder: CaseOps SP8192, 11 layers, 512 hidden dim, 8 query heads / 4 KV heads, XSA on all layers, loop layers 3-5, parallel residual decoder from layer 8, SparseAttnGate, BOS-fixed SmearGate, fused softcapped CE, Polar Express Muon, GPTQ int6 matrices, int7 tied embedding, LQER asymmetric rank-4, per-group lrzip compression, and phased TTT.

## Key Settings

| Setting | Value |
|---------|------:|
| `NUM_LAYERS` | 11 |
| `MODEL_DIM` | 512 |
| `MLP_MULT` | 4.375 |
| Effective MLP blocks | 10 |
| MLP hidden dim | 2240 |
| `QK_GAIN_INIT` | 5.0 |
| `EVAL_SEQ_LEN` | 2048 |
| `TTT_EVAL_SEQ_LEN` | 2048 |
| `MATRIX_LR` | 0.026 |
| `MIN_LR` | 0.1 |
| `WARMDOWN_FRAC` | 0.85 |
| `TTT_LORA_RANK` | 80 |
| `PHASED_TTT_PREFIX_DOCS` | 2500 |
| `GPTQ_RESERVE_SECONDS` | 4.0 |
| `COMPRESSOR` | pergroup |

## Reproduction

```bash
DATA_DIR=./data \
VOCAB_SIZE=8192 \
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
MLP_MULT=4.375 \
EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 TTT_CHUNK_SIZE=48 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
GLOBAL_TTT_MOMENTUM=0.9 WARMDOWN_FRAC=0.85 BETA2=0.99 \
TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 \
GPTQ_RESERVE_SECONDS=4.0 GPTQ_CALIBRATION_BATCHES=16 VAL_LOSS_EVERY=0 \
GATED_ATTN_QUANT_GATE=1 SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 \
SMEAR_GATE_ENABLED=1 \
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Repeat with `SEED=0` and `SEED=1234`.

## Requirements

PyTorch 2.9.1+cu128, CUDA 12.8, 8x H100 80GB SXM, FlashAttention 3, `sentencepiece`, `brotli`, `python-minifier`, and the system `lrzip` binary.

FlashAttention 3 install example:

```bash
pip install --no-deps flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

`lrzip` must be available before running the script.

## Credits

This candidate builds directly on the PR #1855 lineage and keeps its training, quantization, compression, tokenizer, and TTT stack except for the MLP allocation change described above.

Most direct lineage:

- [PR #1855](https://github.com/openai/parameter-golf/pull/1855) by @codemath3000 — BOS-fixed SmearGate + LQER + SparseAttnGate + per-group compression stack.
- [PR #1797](https://github.com/openai/parameter-golf/pull/1797) by @dexhunter — SmearGate + LQER asymmetric rank-4.
- [PR #1787](https://github.com/openai/parameter-golf/pull/1787) by @nprime06 — Polar Express NS, MIN_LR, SparseAttnGate, fused softcapped CE.
- [PR #1736](https://github.com/openai/parameter-golf/pull/1736) — CaseOps + QuantGate + Loop/PhasedTTT integration.
- [PR #1729](https://github.com/openai/parameter-golf/pull/1729) by @romeerp — SP8192 CaseOps tokenizer.
- [PR #1530](https://github.com/openai/parameter-golf/pull/1530) by @samacqua — VarLen attention, fused LeakyReLU-square MLP, parallel residuals, doc-based LoRA TTT.
- [PR #1344](https://github.com/openai/parameter-golf/pull/1344) — Polar Express Newton-Schulz coefficients and depth recurrence.
- [PR #493](https://github.com/openai/parameter-golf/pull/493) — LeakyReLU-square MLP activation.
- [PR #478](https://github.com/openai/parameter-golf/pull/478) — XSA-all.
- [PR #315](https://github.com/openai/parameter-golf/pull/315) — Partial RoPE and LN scale.
- [PR #289](https://github.com/openai/parameter-golf/pull/289) — U-Net skip connections.
