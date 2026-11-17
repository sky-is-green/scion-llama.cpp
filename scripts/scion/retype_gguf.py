#!/usr/bin/env python3
"""Rewrite the ggml type id of selected tensors in a GGUF file.

This is a header-only migration: tensor data is copied byte-for-byte.  It is
only valid when the old and new types have the same block size and byte layout
(the 35B legacy type 43 and PQ2_0 type 142 are both 34 bytes per 128 weights).

Usage:
  python scripts/scion/retype_gguf.py --from 43 --to 142 \
      --in Scion-35B-A3B-PQ2_0-corr.gguf --out Scion-35B-A3B-PQ2_0-corr-v1.gguf
"""

import argparse
import shutil
import struct
import sys


def parse_header(f):
    def u32():
        return struct.unpack("<I", f.read(4))[0]

    def u64():
        return struct.unpack("<Q", f.read(8))[0]

    def s():
        n = u64()
        return f.read(n).decode("utf-8", "replace")

    magic = u32()
    version = u32()
    n_tensors = u64()
    n_kv = u64()
    if magic != 0x46554747 or version != 3:
        raise SystemExit(f"not a GGUF v3 file (magic={magic:#x}, version={version})")

    def skip_value(t):
        if t in (0, 1):
            f.read(1)
        elif t in (2, 3):
            f.read(2)
        elif t in (4, 5, 6):
            f.read(4)
        elif t == 7:
            f.read(1)
        elif t == 8:
            s()
        elif t == 9:
            elem_type = u32()
            n = u64()
            for _ in range(n):
                skip_value(elem_type)
        elif t in (10, 11, 12):
            f.read(8)
        else:
            raise SystemExit(f"unknown GGUF value type {t}")

    for _ in range(n_kv):
        s()
        t = u32()
        if t == 8:
            s()
        elif t in (0, 1, 2, 3, 4, 5, 10, 11):
            fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}[t]
            f.read(struct.calcsize(fmt))
        else:
            skip_value(t)

    tensors = []
    for _ in range(n_tensors):
        name = s()
        n_dims = u32()
        ne = [u64() for _ in range(n_dims)]
        type_pos = f.tell()
        type_id = u32()
        offset = u64()
        tensors.append((name, type_id, type_pos, ne, offset))
    return tensors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--from", dest="from_type", type=int, required=True)
    ap.add_argument("--to", dest="to_type", type=int, required=True)
    args = ap.parse_args()

    with open(args.src, "rb") as f:
        tensors = parse_header(f)

    matches = [(name, pos) for name, ty, pos, _, _ in tensors if ty == args.from_type]
    print(f"{len(tensors)} tensors, {len(matches)} with type {args.from_type}")
    if not matches:
        print("nothing to rewrite", file=sys.stderr)
        return 1

    shutil.copyfile(args.src, args.dst)
    with open(args.dst, "r+b") as g:
        for _, pos in matches:
            g.seek(pos)
            g.write(struct.pack("<I", args.to_type))

    print(f"rewrote {len(matches)} type fields {args.from_type} -> {args.to_type} in {args.dst}")
    print("tensor data was copied unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
