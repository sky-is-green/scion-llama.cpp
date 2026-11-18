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

## Decode attribution (2026-11-18, supersedes earlier guesses)

All numbers measured on this box with the display loaded.

- **Not bandwidth-bound at batch 1.** Cold-cache GEMV (`tools/scion-gemv`, real 2560x10240
  shape, 8-24 distinct copies): Q6_K 299 GB/s, Q5_K 311, Q8_0 336, Q4_0 217, F16 426.
  F16 is 123 us against Q6_K's 72 us because it reads 2.4x the bytes. The model's own
  effective rate is 94 GB/s, so the gap is not in the dot product.
- **The two-card layer split is the largest single cost.** Same 35B weights, same session:
  78.05 tok/s (12.81 ms) on one card, 47.35 (21.12 ms) on two. The penalty rises with the
  display card's share: `-ts` 0.9/0.1 15.59 ms, 0.75/0.25 17.07, 0.6/0.4 21.28, 0.5/0.5
  21.12. Both cards have a 280 W cap, both reach 1249 MHz memory clock, and their sclk
  alternates during decode: the cards execute strictly one after the other.
- **More tokens per step amortise it.** 176B decode at 1/2/4 tokens per step: 42.7/47.0/
  61.9 ms, i.e. 23.4/42.6/64.6 tok/s aggregate. The step has ~35 ms that does not depend
  on the token count.
- **Kernel count**: the 1-token graph is 4138 nodes, 929 matmuls. The other 3209 are
  elementwise/norm/activation/copy (560 ADD, 448 SCALE, 372 MUL, 256 RMS_NORM, 230
  SIGMOID, 151 GET_ROWS, 150 CPY, 134 SILU). `GGML_CUDA_GRAPH_OPT=1` only fuses the
  gate/up GLU pair and gives 21.68 against 21.99 tok/s.
- **The PLE gather is already prefetched.** 48 scattered 90-byte reads into the 26.8 GiB
  Q4_0 `per_layer_token_embd` (zstd btrfs mmap) cost 7.01 ms median when done serially, but
  `llama_prefetch_rows` already issues `madvise(MADV_WILLNEED)` for every row first.
  Forcing the rows in from 8 threads as well gave 31.7 tok/s against 32.7 without it.
- **Parallel split modes**: `-sm row` is unavailable (`device ROCm0 does not support split
  buffers`; the HIP backend has no split buffer type). `-sm tensor` works after the three
  fixes above.
- **Speculative decoding with the current drafter loses.** `draft-mtp` +
  `scion-mtp-sidecar-v2.gguf`: 11.1% acceptance at k=2 (18.5 tok/s) and 5.9% at k=4
  (15.2 tok/s), against 22.0 plain.

## Tensor-parallel split (2026-11-18)

`-sm tensor` is usable for qwen4exp after three fixes:

1. **Loader**: `ggml_backend_meta_buffer_set_tensor` requires offset and size to be
   multiples of `tensor->nb[axis + 1]`. The staged copy fed it fixed 32 MiB pieces and hit
   `GGML_ASSERT(size % chunk_size_full == 0)` on the first tensor. `llama-model-loader.cpp`
   now stages a meta tensor whole (the staging is still what keeps the DMA source resident).
2. **Meta backend**: the Scion correction branches do `mul_mat(mirrored lora_a, AXIS_0
   activation)` and the meta backend aborted with "unsupported mul_mat split states". That
   is a column-parallel matmul whose per-device results are partial sums; it now returns
   PARTIAL, which the backend already all-reduces.
3. **Guard**: `gpu_run.sh` waited for per-card `model buffer size` lines, which a meta
   buffer never prints, so the display cap was never checked for these runs. It now also
   accepts a VRAM plateau and enforces the cap continuously for the whole run.

176B, 358-token prompt, greedy:

| layout | prefill | decode | spare | display |
|---|---:|---:|---:|---:|
| layer 0.55/0.45 | 231 t/s | 22.0 tok/s | 15.5 GiB | 12.6 GiB |
| tensor 0.5/0.5 | 294 t/s | 31.7 tok/s | 16.2 GiB | 18.0 GiB |
| tensor 0.57/0.43 | 221 t/s | 32.8 tok/s | 16.5 GiB | 17.4 GiB |
| tensor 0.6/0.4 | 216 t/s | 32.3 tok/s | 19.0 GiB | 14.9 GiB |

The layer row is the loader's model buffer; the tensor rows are sysfs peaks for the whole
run (model + compute + desktop).

`0.65/0.35` runs the spare out of memory. The usable point is `0.6/0.4`; the display card
is the one with the hard cap, and it is the one this split protects.

Numerics: not bit-identical to the layer split. Each device sums half of the reduction
axis, so greedy decoding follows the same text for ~600 characters and then picks an
equivalent wording. Both were checked against the layer-split output on the same prompt
and seed.

## PLE prefetch (negative result, 2026-11-18)

`per_layer_token_embd` is a 26.8 GiB Q4_0 table on the CPU, read as ~48 scattered 90-byte
rows per token. Serial preads of those rows cost 7.0 ms median, but the model already calls
`llama_prefetch_rows` (which issues `madvise(MADV_WILLNEED)` for all rows) before the
gather. Forcing the rows in from 8 threads as well gave 31.7 tok/s against 32.7 without it,
so the existing prefetch already overlaps the reads and the 7.0 ms figure is not the
model's cost.

## Measured limits (2026-11-18)

- **All-GPU 176B works**: `-ncmoe 0 -ts 0.5,0.5` loads (1228 tensors, ~2.5 min) and runs **prefill 200 t/s, decode 22.0 t/s** (358-token prompt, 64 generated). A 32k-token prompt is ~2.7 min. A short 57-token prompt reports 27 t/s prefill because the fixed per-request overhead dominates.
- **Why the all-GPU load used to hang**: copying a device tensor straight out of the mmap wedges the KFD on gfx1100 + ROCm 7.2.4. It reproducibly stopped on `blk.41.ffn_up_exps.weight` (175 MiB to the display card) with no disk I/O, no VRAM change, two threads in `kfd_wait_on_events`, and `HSA_ENABLE_SDMA=0` made no difference. The DMA engine faults the mmap pages in itself and the wait never returns. `SCION_LOAD_STAGE_MIB` (default 32) stages the copy through resident memory in chunks.
- **Host-expert layouts** (`-ncmoe 24`, 9 GiB MoE cache) are 50-400x slower than all-GPU: prefill 0.25-0.56 t/s, decode 0.04-0.14 t/s, with 27 CPU↔GPU splits per decode step and most time in per-split fence waits.
- **`TQ1_0_G128` MMQ**: correct (`test-backend-ops` 13/13) but neutral for this workload; decode uses MMVQ. `SCION_DISABLE_TQ1_MMQ=1` turns it off.
- **MoE cache policy**: per-layer slot banks (as in the Prism fork's `--moe-slot-cache`) are implemented. On the 176B they change nothing: decode hit rate is 58.52% and uploaded bytes are identical with either the global LRU or per-layer banks, so the global pool was not contended. `--moe-cache-mib 9000` gives 365 slots per layer (71% of each layer's experts, ~36 tokens of history).
- **PCIe is not the limit**: 13.6 GB/s host→device from both resident memory and the compressed model mmap (256 MB probe). Fence waits on an idle stream are 0.019 ms.
- **CPU expert compute is not the limit**: with `--moe-cache-cpu-share 1.0` the log reports `0 on the CPU`; MMQ support makes `ggml_backend_offload_op(MUL_MAT_ID)` true, which disables the CPU-share path entirely, so all misses are uploaded and computed on the device.

## Known limits

- **176B speed on this host**: use the fully-resident GPU layout: `-ncmoe 0 -ts 0.5,0.5 --moe-cache-mib 0 -ot per_layer_token_embd=CPU`. It loads in ~2.5 min and runs prefill 200 t/s, decode 22 t/s.
- **176B prefill**: 200 t/s all-GPU; 0.25-0.56 t/s whenever experts are host-resident.
- **Display load**: these are desktop-loaded numbers. A full-screen game on card1 is not the same thing and cuts the 35B to 17.4 tok/s and the 176B by ~100x; desktop compositing and video playback are part of the stated operating condition and are included in every number here.
- **176B sidecar acceptance**: the fc3 chain is implemented, but the release's offline 1.81 tokens/step has not yet been re-measured in-engine on v1.
- **ROCm 10.1**: the 2026-11-17 system update broke `hipMemcpyAsync` for every HIP binary (including stock v1.0.0 and the system llama.cpp). The 7.2.4 runtime is extracted under `work/rocm-7.2.4` and selected with `LD_LIBRARY_PATH`; this is a host workaround, not a port requirement.
- **Backends**: CPU + HIP (gfx1100) only. No Metal/Vulkan validation.
