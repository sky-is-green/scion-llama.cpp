# scion-llama.cpp

Scion's llama.cpp. Based on upstream [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)
tag **v1.0.0** (`b1b770d90`), branch `scion-base`. This fork exists only to run the Scion
models; it is not a general-purpose llama.cpp distribution and does not track upstream.

MIT licensed, same as upstream.

## Models

| model | file | expert container |
|---|---|---|
| Scion 35B A3B | `Scion-35B-A3B-Q2_0_G128-corr-v1.gguf` | `Q2_0_G128` |
| Scion 35B A3B base-router | `Scion-35B-A3B-Q2_0_G128-base-router-v1.gguf` | `Q2_0_G128` |
| Scion FlashNext 176B A6B | `Scion-FlashNext-176B-A6B-Q6_K.gguf` | `TQ1_0_G128` |
| Scion FlashNext 176B A6B | `Scion-FlashNext-176B-A6B-Q6_K-Q4X160.gguf` | `TQ1_0_G128` + `Q4X160` PLE |

## Container types

| id | name | layout |
|---:|---|---|
| 142 | `Q2_0_G128` | 128 weights per block, 2-bit, group 128 |
| 143 | `TQ1_0_G128` | 128 weights per block, ternary, 5 trits per byte + `qh` |
| 146 | `Q4X160` | 160 weights per row, 16-level codebook |

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
./build-hip/bin/llama-cli -m Scion-FlashNext-176B-A6B-Q6_K.gguf \
  --device ROCm0,ROCm1 -ts 0.5,0.5 -ngl 99 -ncmoe 0 --moe-cache-mib 0 \
  -ot per_layer_token_embd=CPU -c 4096 -fit off \
  -p "The capital of France is" -n 16 -st --temp 0 --top-k 1 --top-p 1 -s 1001
```

Load takes ~2.5 min. The display card must stay under 18 GiB total (model + desktop);
`scripts/scion/gpu_run.sh` enforces it at runtime and `scripts/scion/gpu_guard.sh` checks
the flags before launch. Use both for every GPU run.

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

### 35B A3B, all GPU

| measurement | value |
|---|---:|
| decode, clean box | 75.9 tok/s |
| decode, with desktop GPU load | 49.5 tok/s |
| tokens identical to the Prism fork | 16/16 |
| drafter acceptance | 71.4% at 69.1 tok/s |

### Kernel validation

`test-backend-ops -b ROCm0 -o MUL_MAT,MUL_MAT_ID` for `Q2_0_G128` and `TQ1_0_G128`:
**26/26**, including the MoE shapes.

## Results log

New measurements go here, newest first, with the command and the machine state.

- **2026-11-18** — all-GPU 176B: 200 tok/s prefill, 22.0 tok/s decode, token parity with
  the host-split path. Blocked until the loader stopped copying device tensors straight
  from the mmap; that copy wedged in `kfd_wait_on_events` on `blk.41.ffn_up_exps.weight`
  (175 MiB to the display card) with no I/O and no VRAM change. `HSA_ENABLE_SDMA=0` did
  not help, `SCION_LOAD_STAGE_MIB=32` did.
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
