# scion-llama.cpp

Scion's llama.cpp. Based on upstream [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)
tag **v1.0.0** (`b1b770d90`), branch `scion-base`. This fork exists only to run the Scion
models; it is not a general-purpose llama.cpp distribution and does not track upstream.

MIT licensed, same as upstream.

## Prebuilt release

[`scion-v1.0.0`](https://github.com/sky-is-green/scion-llama.cpp/releases/tag/scion-v1.0.0)
has Linux x64 ROCm (gfx1100) and CPU tarballs built from commit `2089cd1d3`
(build 12389). The ROCm tarball was smoke-tested on ROCm 7.2.4 / RX 7900 XT
with the 16.2 GB Q4 model at 58.8 tok/s. No Windows, macOS or CUDA binaries are
published yet; build those from source.

The older Prism `moe-corr-runtime` fork remains a legacy backup for the Q2
model only; it predates `Q4_0_G128` (type 147) and cannot load the Q4 variant.

## Models

| model | file | expert container |
|---|---|---|
| Scion 35B A3B | `Scion-35B-A3B-Q2_0_G128-corr-v1.gguf` | `Q2_0_G128` |
| Scion 35B A3B base-router | `Scion-35B-A3B-Q2_0_G128-base-router-v1.gguf` | `Q2_0_G128` |
| Scion 35B A3B Q4 gate/up | `Scion-35B-A3B-Q4_0-GateUp-TQ1Down-v1corr.gguf` | `Q4_0_G128` gate/up + `TQ1_0_G128` down |
| Scion FlashNext 176B A6B | `Scion-FlashNext-176B-A6B-Q6_K.gguf` | `TQ1_0_G128` |
| Scion FlashNext 176B A6B | `Scion-FlashNext-176B-A6B-Q6_K-Q4X160.gguf` | `TQ1_0_G128` + `Q4X160` PLE |

## Container types

| id | name | layout |
|---:|---|---|
| 142 | `Q2_0_G128` | 128 weights per block, 2-bit, group 128 |
| 143 | `TQ1_0_G128` | 128 weights per block, ternary, 5 trits per byte + `qh` |
| 146 | `Q4X160` | 160 weights per row, 16-level codebook |
| 147 | `Q4_0_G128` | 128 weights per block, 4-bit, group 128 |

The Prism fork called these `PQ2_0` / `PTQ1_0`. The IDs and tensor bytes are unchanged;
only the public names differ.

## What this fork adds

- **Container types 142/143/146** with reference quantizers and dequantizers, CPU
  `vec_dot`, CUDA/HIP dequant/get_rows/MMVQ.
- **`TQ1_0_G128` MMQ on RDNA3** — tile loader, RDNA3 configs, DP4A path. `SCION_DISABLE_TQ1_MMQ=1`
  forces the old path.
- **Embedded adapters** — `.lora_a/.lora_b` stored in the model file, applied automatically.
- **TAARDIS correction hook** — `blk.N.ffn_moe_out.weight` applied at the MoE block output
  through `build_lora_branch`.
- **MTP drafter lanes** — direct `draft-mtp-sidecar` (35B), `mtp-sidecar` + fc3 chain (176B),
  `n_embd_nextn` split.
- **Per-layer MoE cache banks** — each host-resident MoE layer owns a slice of the expert
  cache (as in the Prism `--moe-slot-cache`), instead of one shared LRU.
- **Staged weight copies** — device tensors are copied through resident memory in chunks
  (`SCION_LOAD_STAGE_MIB`, default 32). Required to load the all-GPU 176B on this runtime;
  see the results log.

## Build (ROCm, gfx1100)

ROCm 10.1 as shipped broke hipMemcpy for every HIP binary on this box. An extracted 7.2.4
runtime is used at build-run time via `LD_LIBRARY_PATH`; do not upgrade HIP without testing.

```bash
cmake -S . -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j 12
```

## Running the 176B (all GPU)

Everything that is not the PLE table fits in the two cards; the PLE stays host-mapped.
`HIP_VISIBLE_DEVICES=1,0` puts the spare card first, so `-ts 0.5,0.5` splits evenly.

```bash
export LD_LIBRARY_PATH=$W/rocm-7.2.4/opt/rocm/lib:$W/rocm-7.2.4/usr/lib
export HIP_VISIBLE_DEVICES=1,0

# fast: both cards work on every token (tensor parallel, 32 tok/s)
./build-hip/bin/llama-cli -m Scion-FlashNext-176B-A6B-Q6_K.gguf \
  --device ROCm0,ROCm1 -sm tensor -ts 0.6,0.4 -ngl 99 -ncmoe 0 --moe-cache-mib 0 \
  -ot per_layer_token_embd=CPU -c 4096 -fit off \
  -p "The capital of France is" -n 16 -st --temp 0 --top-k 1 --top-p 1 -s 1001

# conservative: half the layers per card (22 tok/s), 4 GiB more spare VRAM
./build-hip/bin/llama-cli -m Scion-FlashNext-176B-A6B-Q6_K.gguf \
  --device ROCm0,ROCm1 -sm layer -ts 0.55,0.45 -ngl 99 -ncmoe 0 --moe-cache-mib 0 \
  -ot per_layer_token_embd=CPU -c 4096 -fit off \
  -p "The capital of France is" -n 16 -st --temp 0 --top-k 1 --top-p 1 -s 1001
```

The tensor split is the fast path and keeps the display card at ~15 GiB, but the spare
ends up at ~19.0 of 20.0 GiB, so use the layer split when you need spare headroom (for a
longer context). `-ts 0.6,0.4` is the balanced point: `0.65,0.35` runs the spare out of
memory and `0.5,0.5` puts the display at 18.0 GiB. Load takes ~2.5 min. The display card
must stay under 18 GiB total (model + desktop); `scripts/scion/gpu_run.sh` now enforces it
for the whole run, and `scripts/scion/gpu_guard.sh` checks the flags before launch. Use
both for every GPU run.

## Results

Machine: 2x RX 7900 XT (20 GiB each), Ryzen 7 7800X3D, 30 GiB RAM, NVMe/btrfs.

### 176B Q6_K, all GPU (2026-11-18)

`-ncmoe 0 -ts 0.5,0.5 --moe-cache-mib 0`, 358-token prompt, 64 generated:

| metric | value |
|---|---:|
| prefill | **200 tok/s** |
| decode | **22.0 tok/s** |
| load | ~2.5 min |
| VRAM | 14.7 GiB spare + 15.2 GiB display |

Output matches the host-split configuration on the same prompt and seed.

### 176B Q6_K, host experts (reference, do not use)

`-ncmoe 24 --moe-cache-mib 9000 --moe-cache-cpu-share 1.0`: prefill 0.25-0.56 tok/s,
decode 0.04-0.14 tok/s. 27 CPU/GPU splits per decode step dominate; kept only for debugging.

### Decode attribution (2026-11-18)

The 176B is **not** bandwidth-bound at batch 1. Per-token bytes are ~4.26 GB
(3.74 dense + 0.52 active experts), but a cold-cache microbenchmark of the actual
weight shapes shows the kernel is not the limit:

`scion-gemv-bench -k 2560 -m 10240` (8-24 distinct cold copies, f32 activations):

| type | bytes/call | time | GB/s |
|---|---:|---:|---:|
| `Q6_K` | 26.8 MB | 72 µs | 299 |
| `Q5_K` | 22.5 MB | 58 µs | 311 |
| `Q8_0` | 34.8 MB | 83 µs | 336 |
| `Q4_0` | 18.4 MB | 68 µs | 217 |
| `F16` | 52.4 MB | 123 µs | 426 |

F16 wins on GB/s and loses on wall time: 2.4x the bytes makes it 1.7x slower than
Q6_K. That closes the "repack to F16 and use rocBLAS" idea.

Where the time actually goes, all measured on this box:

| observation | result |
|---|---|
| 176B effective bandwidth | 94 GB/s (vs 299 isolated) |
| 35B, one card | 78.1 tok/s, 12.81 ms, 231 GB/s |
| 35B, two cards 0.5/0.5 | 47.4 tok/s, 21.12 ms, 140 GB/s |
| same model, `-ts` 0.9/0.1 / 0.75/0.25 / 0.6/0.4 | 15.6 / 17.1 / 21.3 ms |
| 176B, tokens per step 1 / 2 / 4 | 42.7 / 47.0 / 61.9 ms |
| 1-token graph | 4138 kernels, only 929 matmuls |
| PLE serial preads | 7.0 ms (the existing prefetch hides most of it) |

Consequences:

- **The layer split is the dominant tax.** The same 35B weights run 1.65x slower
  across two cards, and the penalty grows with the share given to the display card
  (~0.35 ms per layer moved). Both cards sit at their 280 W cap and reach 1249 MHz
  memory clock, and the clocks alternate — the cards run strictly one at a time.
  A second token per step costs only 10%, which is the same fact from the other
  side: the step has a large fixed cost that more tokens amortize.
- **`-sm row` cannot be used**: the HIP backend has no split-buffer type
  (`device ROCm0 does not support split buffers`). `-sm tensor` works after the loader,
  meta-backend and guard fixes described under Tensor-parallel split below.
- **3209 of 4138 kernels are not matmuls** (560 ADD, 448 SCALE, 372 MUL, 256
  RMS_NORM, 230 SIGMOID, 151 GET_ROWS, 150 CPY, 134 SILU, ...). Upstream's
  `GGML_CUDA_GRAPH_OPT=1`, which only fuses the gate/up GLU pair, gives
  21.68 tok/s against 21.99 — no help.
- **The PLE gather is not the stall the raw number suggests.** The 26.8 GiB Q4_0
  `per_layer_token_embd` lives on the CPU as a zstd btrfs mmap, and a token reads 48
  scattered 90-byte rows; serial preads of those rows cost 7.0 ms median. The model
  already calls `llama_prefetch_rows`, which issues `madvise(MADV_WILLNEED)` for every
  row before the gather, and forcing them in from 8 threads as well gave 31.7 tok/s
  against 32.7 without it.
- **Speculative decoding does not pay with this drafter**: `draft-mtp` with
  `scion-mtp-sidecar-v2.gguf` accepts 11.1% at k=2 and 5.9% at k=4, for 18.5 and
  15.2 tok/s against 22.0 plain.

### Tensor-parallel split (2026-11-18)

`-sm tensor` puts a slice of every weight matrix on each card instead of half the layers,
so both cards work on every token. On this fork it needed three fixes:

- the staged loader fed the meta buffer fixed 32 MiB pieces, but that buffer requires whole
  slabs (`size % nb[axis + 1] == 0`), so it now stages a meta tensor whole;
- the Scion correction branches do `mul_mat(mirrored lora_a, row-parallel activation)`,
  which the meta backend did not implement; it now produces a PARTIAL sum that is
  all-reduced like the other column-parallel cases;
- `gpu_run.sh` only checked the display cap when the loader printed per-card buffer sizes,
  which a meta buffer never does; it now also accepts a VRAM plateau and keeps enforcing
  the cap for the whole run.

176B, 358-token prompt, greedy, display card also running the desktop:

| layout | prefill | decode | spare | display |
|---|---:|---:|---:|---:|
| layer 0.55/0.45 | 231 t/s | 22.0 tok/s | 15.5 GiB | 12.6 GiB |
| tensor 0.5/0.5 | 294 t/s | 31.7 tok/s | 16.2 GiB | **18.0 GiB** |
| tensor 0.57/0.43 | 221 t/s | 32.8 tok/s | 16.5 GiB | 17.4 GiB |
| tensor 0.6/0.4 | 216 t/s | 32.3 tok/s | **19.0 GiB** | 14.9 GiB |

The layer row is the loader's model buffer; the tensor rows are sysfs peaks for the whole
run (model + compute + desktop), which is the number the 18 GiB rule is about.

**Use `-ts 0.6,0.4`**: it is the only split that keeps the display card comfortably under
its 18 GiB cap, at the cost of the spare sitting at 19.0 of 20.0 GiB. `0.65,0.35` runs the
spare out of memory. The split is steep between 0.4 and 0.43 because a whole tensor (or a
device-sized allocation) moves at that step, so do not interpolate these numbers.

The output is not bit-identical to the layer split: each device sums half of the reduction
axis, so greedy decoding follows the same text for ~600 characters and then picks an
equivalent wording. For the 35B, which fits on one card, tensor split is slower than one
card (66.5 vs 78.1 tok/s) and only beats the layer split (47.4).

### 35B A3B, all GPU

| measurement | value |
|---|---:|
| decode, one card | 78.1 tok/s |
| decode, two cards 0.5/0.5 | 47.4 tok/s |
| decode with desktop GPU load | 49.5 tok/s |
| tokens identical to the Prism fork | 16/16 |
| drafter acceptance | 71.4% at 69.1 tok/s |

### Kernel validation

`test-backend-ops -b ROCm0 -o MUL_MAT,MUL_MAT_ID` for `Q2_0_G128` and `TQ1_0_G128`:
**26/26**, including the MoE shapes.

## Results log

New measurements go here, newest first, with the command and the machine state.

- **2026-11-18** — `-ub 256` is not the lever for the spare's VRAM. Spare peak 19324 MiB
  against 19410 at the default 512, display 15151 against 15239, while prefill drops from
  216 to 176 t/s and decode is unchanged within noise (33.4 vs 32.3). The meta compute
  buffer is 78.5 MiB, so the spare's ~19 GiB is the resident split of the weights (0.6 of
  the split tensors plus the whole mirrored set), not a ubatch-sized spike. Keep the
  default ubatch.
- **2026-11-18** — tensor-parallel split works for qwen4exp: 176B decode **22.0 → 32.3 tok/s**
  (`-sm tensor -ts 0.6,0.4`), prefill 216-294 tok/s, at spare 19.0 / display 14.9 GiB.
  Three fixes were needed: the staged loader could not feed the meta buffer its required
  whole slabs, the meta backend had no `mul_mat(MIRRORED weight, AXIS_0 activation)` case
  for the Scion correction branches (now a PARTIAL sum), and `gpu_run.sh` skipped the
  display cap because a meta buffer prints no per-card report (now enforced continuously,
  from a VRAM plateau when there is no report). Output is not bit-identical: greedy text
  matches for ~600 characters, then takes an equivalent wording, because each device sums
  half the reduction axis. 35B: 47.4 → 66.5 tok/s, still below 78.1 on one card.
- **2026-11-18** — PLE prefetch is already good: forcing the 48 scattered rows in from 8
  threads (`SCION_PLE_TOUCH=1`) gave 31.7 tok/s against 32.7 without it, so the existing
  `madvise(MADV_WILLNEED)` in `llama_prefetch_rows` already overlaps the reads. The earlier
  7.0 ms figure measured serial preads without that prefetch; it is not the model's cost.
- **2026-11-18** — `-ts 0.55,0.45` replaces 0.5/0.5 as the default: 22.37 tok/s (44.71 ms)
  against 21.99 (45.48), and the display card's model buffer drops from 15.2 to 12.85 GiB.
  Same speed within run-to-run variance, 2.3 GiB more headroom under the 18 GiB cap.
- **2026-11-18** — decode attribution, superseding the "weight-bandwidth bound" reading below.
  The dense GEMV is not the limit: cold-cache `scion-gemv-bench` on the real shape gives
  299 GB/s for Q6_K, 336 for Q8_0, 311 for Q5_K, 426 for F16 (F16 loses on wall time:
  123 µs vs 72 µs). In the model the 176B reaches only 94 GB/s. The gap is the two-card
  layer split: the same 35B runs 78.1 tok/s on one card (231 GB/s) and 47.4 tok/s on two
  (140 GB/s), with the penalty scaling at ~0.35 ms per layer moved to the display card.
  Both cards are at their 280 W cap, reach 1249 MHz memory clock, and their clocks
  alternate, so they run one at a time. `-sm row` is refused (no HIP split buffers);
  `-sm tensor` aborts after the `Meta()` buffer allocation. Q6_K nwarps 2 -> 8 changes
  nothing (45.47 vs 45.48 ms/token). `GGML_CUDA_GRAPH_OPT=1` gives 21.68 vs 21.99 tok/s.
  The 1-token graph is 4138 kernels, only 929 of them matmuls. `per_layer_token_embd`
  gather costs 7.0 ms/token on the CPU (48 scattered 90-byte rows from a 26.8 GiB
  zstd-mapped Q4_0 table). The 176B `draft-mtp` sidecar accepts 11.1% at k=2 (18.5 tok/s)
  and 5.9% at k=4 (15.2 tok/s), both worse than 22.0 plain.
- **2026-11-18** — batching shows the step is fixed-cost dominated: 176B decode at 1, 2 and
  4 tokens per step is 42.7, 47.0 and 61.9 ms/step (23.4, 42.6 and 64.6 tok/s aggregate).
  Four times the work for 45% more time.
- **2026-11-18** — `scion-gemv-bench` added (`tools/scion-gemv`): cold-cache batch-1 GEMV
  timing for one shape, rotating through `-r` distinct copies so the 96 MB Infinity Cache
  does not hold the matrix. `test-backend-ops perf` shapes fit in cache and report
  ~1.8 TB/s, which is why they cannot be used for this.
- **2026-11-18** — all-GPU 176B: 200 tok/s prefill, 22.0 tok/s decode, token parity with
  the host-split path. Blocked until the loader stopped copying device tensors straight
  from the mmap; that copy wedged in `kfd_wait_on_events` on `blk.41.ffn_up_exps.weight`
  (175 MiB to the display card) with no I/O and no VRAM change. `HSA_ENABLE_SDMA=0` did
  not help, `SCION_LOAD_STAGE_MIB=32` did.
- **2026-11-18** — decode attribution on the all-GPU 176B: prefill 861 tok/s at a 5270-token
  prompt; decode 21.6 tok/s at 5.3k context vs 19.2 at 57 tokens, so decode is flat in
  context and weight-bandwidth bound. `GGML_CUDA_DISABLE_GRAPHS=1` gives 20.0 tok/s, so HIP
  graphs are not a lever. Per token the model reads ~4.25 GB (3.74 GB dense, 0.52 GB active
  experts): decode is dense-bound, not expert-bound.
- **2026-11-18** — per-layer MoE banks: 365 slots/layer at 9 GiB, decode hit rate 58.52%,
  identical to the global LRU. The pool was not contended; no speedup.
- **2026-11-17** — `TQ1_0_G128` MMQ: 13/13 against the CPU reference; neutral for the 176B
  because decode uses MMVQ.
- **2026-11-17** — 176B sidecar output identical to the Prism engine; 50% acceptance on a
  two-draft sample. 35B drafter 71.4% / 69.1 tok/s.
- **pre-2026-11-17** — 35B GPU 75.9 tok/s on an idle box; 16/16 tokens identical to the
  Prism fork.

## Layout

- `src/llama-moe-cache.cpp` — expert cache (per-layer banks)
- `src/llama-model-loader.cpp` — staged weight copies, `SCION_TRACE_LOAD` per-tensor trace
- `ggml/src/ggml-cuda/` — type 142/143/146 kernels, MMQ
- `scripts/scion/` — `gpu_guard.sh`, `gpu_run.sh`, `retype_gguf.py`
- `docs/PORT-STATUS.md` — port status and the full measured-limits list
