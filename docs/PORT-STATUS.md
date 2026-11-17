# Scion llama.cpp v1 port — status

Base: upstream `ggml-org/llama.cpp` tag **v1.0.0** (`b1b770d90d94cad9e9ea5b993bdbc383a522fb28`).
Branch: `scion-base`.

## Commits

| commit | scope |
|---|---|
| `94ff9318e` | types 142/143/146, embedded adapters, `ffn_moe_out` correction hook, 43→142 re-export tool |
| `fa2460759` | `draft-mtp-sidecar` + `mtp-sidecar` drafter lanes, `n_embd_nextn` split, CUDA/HIP dequant/get_rows/MMVQ |
| `9d937187e` | fc3 chaining for the 176B `mtp-sidecar` in the `draft-mtp` lane |

## What is ported

- **Q2_0_G128 (142)**, **TQ1_0_G128 (143)**, **Q4X160 (146)**: block layouts, reference quantizers, dequantizers, generic CPU vec_dot, type traits, CPU get_rows, CUDA/HIP dequant/get_rows/MMVQ.
- **TQ1_0_G128 MMQ on RDNA3**: a tile loader that expands the packed base-3 digits into signed bytes in the DP4A layout, tile configs in `mmq-config-rdna3{,-5}`, DS-layout/tile-size/util-func cases, and a generated template instance. The type has no MMA tile loader, so `use_mma_data_layout()` is false and the SRAM sizing follows the DP4A tile. `SCION_DISABLE_TQ1_MMQ=1` forces the old path for A/B runs.
- **Embedded adapters**: `.lora_a/.lora_b` tensors embedded in the model file, `adapter.embedded=true`, `llama_model.internal_lora`, auto-apply in every context.
- **TAARDIS correction hook**: `blk.N.ffn_moe_out.weight` is mapped to the `ffn_gate_inp` anchor and applied through `build_lora_branch` at the MoE block output. Ordinary `ssm_out`, `attn_output`, and `ffn_gate_inp` LoRAs use the existing `build_lora_mm` path.
- **35B drafter**: direct `draft-mtp-sidecar` lane (fc1/fc2, no draft model).
- **176B v2 sidecar**: `mtp-sidecar` model architecture + upstream `draft-mtp` lane, with the fc3 state-head chain.
- **`n_embd_nextn`**: qwen4exp's MTP head input is the 2560-wide final mixer output, distinct from the 10240-wide hc state.
- **Header re-export**: `scripts/scion/retype_gguf.py` rewrites the 35B legacy type 43 to 142 without touching tensor bytes.

## Validation (greedy, same prompt)

| model | CPU | GPU |
|---|---|---|
| 35B corr | 16/16 tokens identical to the Prism fork | 16/16 identical, 75.9 tok/s on the spare |
| 35B base-router | text identical | not run |
| 176B Q6_K (Q4_0 PLE) | 8/8 identical | 8/8 identical; 0.2 tok/s with host experts |
| 176B Q4X160 (type 146) | 8/8 identical | not run on GPU |
| 35B drafter | 71.4% acceptance, output identical | 71.4% acceptance, 69.1 tok/s (baseline 75.9) |
| 176B v2 sidecar | k=1/chain, output identical | 50% on a 2-draft sample; target is the bottleneck |
| `TQ1_0_G128` MMQ kernel | — | `test-backend-ops` `MUL_MAT` + `MUL_MAT_ID` on ROCm0: **13/13** pass |

The 35B subset30 evaluation is running against the v1 server; results will be added when it finishes.

## Known limits

- **176B speed on this host**: the fully-resident GPU layout stalls in the ROCm loader (same as the fork), so 24 layers' experts stay in host RAM. v1's MoE cache defaults to uploading misses to the GPU (`--moe-cache-cpu-share 0`), which makes the 176B decode ~0.2 tok/s. `--moe-cache-cpu-share 1.0` plus `--moe-cache-mib 9000` is the working split: decode 7.45 tok/s at 24 host layers in the one clean measurement, prefill still ~0.55 tok/s.
- **176B prefill**: host-resident expert layers are processed essentially per token, so prompt processing runs at decode-like speed. With 24 host layers that is 0.3-0.6 tok/s regardless of MMQ. Prefill, not decode, is what makes a 30x32k overnight run infeasible on this box.
- **CPU governor / GPU contention**: results are very sensitive to load on the display card. A game using card1 cut the 35B from 75.9 to 17.4 tok/s and the 176B by ~100x; measure only with the display card idle.
- **176B sidecar acceptance**: the fc3 chain is implemented, but the release's offline 1.81 tokens/step has not yet been re-measured in-engine on v1.
- **ROCm 10.1**: the 2026-11-17 system update broke `hipMemcpyAsync` for every HIP binary (including stock v1.0.0 and the system llama.cpp). The 7.2.4 runtime is extracted under `work/rocm-7.2.4` and selected with `LD_LIBRARY_PATH`; this is a host workaround, not a port requirement.
- **Backends**: CPU + HIP (gfx1100) only. No Metal/Vulkan validation.
