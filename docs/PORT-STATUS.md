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

- **PQ2_0 (142)**, **PTQ1_0 (143)**, **Q4X160 (146)**: block layouts, reference quantizers, dequantizers, generic CPU vec_dot, type traits, CPU get_rows, CUDA/HIP dequant/get_rows/MMVQ.
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

The 35B subset30 evaluation is running against the v1 server; results will be added when it finishes.

## Known limits

- **176B speed on this host**: the fully-resident GPU layout stalls in the ROCm loader (same as the fork), so 24 layers' experts stay in host RAM. v1's MoE cache defaults to uploading misses to the GPU (`--moe-cache-cpu-share 0`), which makes the 176B decode ~0.2 tok/s. The next tuning step is `--moe-cache-cpu-share 1.0` plus a larger `--moe-cache-mib`; the fork's slot cache reached 22.5 tok/s with a similar split.
- **176B sidecar acceptance**: the fc3 chain is implemented, but the release's offline 1.81 tokens/step has not yet been re-measured in-engine on v1.
- **ROCm 10.1**: the 2026-11-17 system update broke `hipMemcpyAsync` for every HIP binary (including stock v1.0.0 and the system llama.cpp). The 7.2.4 runtime is extracted under `work/rocm-7.2.4` and selected with `LD_LIBRARY_PATH`; this is a host workaround, not a port requirement.
- **Backends**: CPU + HIP (gfx1100) only. No Metal/Vulkan validation.
