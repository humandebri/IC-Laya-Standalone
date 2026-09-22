#!/usr/bin/env python3
"""Convert an existing canonical F32 pack to per-row symmetric int8.

Matrices become signed int8 row-major bytes followed by one LE F32 scale per
row. Vectors remain F32. The manifest hashes both weights and scales. This is
W8A8 at inference (dynamic per-token activation quantization), not a quality gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from pack_checkpoint import validate_config, MAX_TENSOR, MAX_TOTAL
from model_reference import shapes


def quantize(source: Path, out: Path):
    if out.exists():
        raise ValueError("output already exists")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["format"] != "ic-laya-f32-pack-v1":
        raise ValueError("source must be a canonical F32 pack")
    validate_config(manifest["config"])
    expected = shapes(manifest["config"])
    if len(manifest["tensors"]) != len(expected) or {e["name"] for e in manifest["tensors"]} != set(expected):
        raise ValueError("tensor set mismatch")
    if not 0 < manifest["total_bytes"] <= MAX_TOTAL or (source / "model.bin").stat().st_size != manifest["total_bytes"]:
        raise ValueError("model length mismatch")
    tok = (source / "tokenizer.json").read_bytes()
    if list(hashlib.sha256(tok).digest()) != manifest["tokenizer_sha256"]:
        raise ValueError("tokenizer integrity")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".int8-pack-", dir=out.parent))
    try:
        entries = []
        offset = 0
        source_offset = 0
        with (source / "model.bin").open("rb") as src, (tmp / "model.bin").open("wb") as dst:
            for entry in manifest["tensors"]:
                shape = expected[entry["name"]]
                length = int(np.prod(shape)) * 4
                if list(shape) != entry["shape"] or entry["offset"] != source_offset or entry["length"] != length or length > MAX_TENSOR:
                    raise ValueError("tensor shape/offset/length")
                raw = src.read(length)
                if list(hashlib.sha256(raw).digest()) != entry["sha256"]:
                    raise ValueError("tensor integrity")
                values = np.frombuffer(raw, dtype="<f4").reshape(shape)
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite weight")
                storage = "F32"
                if len(shape) == 2:
                    peak = np.max(np.abs(values), axis=1)
                    scales = np.where(peak == 0, 1., np.maximum(peak / np.float32(127), np.finfo(np.float32).tiny)).astype("<f4")
                    # Half away from zero, matching Rust's f32::round.
                    scaled = values / scales[:, None]
                    quantized = np.copysign(np.floor(np.abs(scaled) + np.float32(.5)), scaled).clip(-127, 127).astype(np.int8)
                    raw = quantized.tobytes() + scales.tobytes()
                    storage = "I8Row"
                dst.write(raw)
                entries.append(dict(entry, storage=storage, offset=offset, length=len(raw), sha256=list(hashlib.sha256(raw).digest())))
                offset += len(raw)
                source_offset += length
        if source_offset != manifest["total_bytes"]:
            raise ValueError("source total length")
        manifest.update(format="ic-laya-int8-pack-v1", total_bytes=offset, tensors=entries)
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (tmp / "tokenizer.json").write_bytes(tok)
        for name in ("input.json", "cases.json", "export_mapping.json", "export_config.json"):
            if (source / name).exists():
                shutil.copyfile(source / name, tmp / name)
        tmp.rename(out)
        return manifest
    except Exception:
        shutil.rmtree(tmp)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result = quantize(args.source, args.out)
    print(json.dumps({"format": result["format"], "bytes": result["total_bytes"], "quality_verified": False}))
