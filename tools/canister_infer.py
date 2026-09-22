#!/usr/bin/env python3
"""Upload a Laya pack and run token-level inference on this project's local IC.

Uses 1 MiB binary Candid files, so real checkpoints do not hit argv limits.
Never reinstalls a canister. --initialize installs only on a fresh canister;
use the explicit icp upgrade workflow for an existing installed canister.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct
import tempfile
import time
from measure_inference import (Icp, ROOT, BUILD, Failure, blob, principal,
                               require_local_network, network_status,
                               ensure_identity, ensure_cycles, ensure_canister)


def uleb(value):
    out = bytearray()
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def decode_blobs(text):
    values = []
    for body in re.findall(r'blob\s+"((?:[^"\\]|\\.)*)"', text):
        value = bytearray()
        i = 0
        escapes = {"n": 10, "r": 13, "t": 9, '"': 34, "'": 39, "\\": 92}
        while i < len(body):
            if body[i] != "\\":
                value.extend(body[i].encode())
                i += 1
            elif body[i+1] in escapes:
                value.append(escapes[body[i+1]])
                i += 2
            else:
                value.append(int(body[i+1:i+3], 16))
                i += 3
        values.append(bytes(value))
    return values


def upload(icp, directory):
    raw = (directory / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    tok = (directory / "tokenizer.json").read_bytes()
    if hashlib.sha256(tok).digest() != bytes(manifest["tokenizer_sha256"]):
        raise Failure("tokenizer hash mismatch")
    tokens = {v["content"]: v["id"] for v in json.loads(tok)["added_tokens"]}
    names = {"cls": "[CLS]", "sep": "[SEP]", "mask": "[MASK]", "pad": "[PAD]"}
    special = "; ".join(f"{field} = {tokens[name]} : nat32" for field, name in names.items())
    literals = "; ".join(json.dumps(name) for name in names.values())
    result = icp.call("decision-engine", "begin_upload", f"({blob(raw)}, {len(tok)} : nat64, record {{ {special}; literals = vec {{ {literals} }} }})")
    if "Ok" not in result:
        raise Failure(result)
    offset = 0
    with tempfile.TemporaryDirectory(prefix="laya-upload-") as temp:
        argsfile = Path(temp) / "args.bin"
        for path in [directory / "model.bin", directory / "tokenizer.json"]:
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    # Type table: vec nat8. Args: nat64, type-table entry 0.
                    argsfile.write_bytes(b"DIDL\x01\x6d\x7b\x02\x78\x00" + struct.pack("<Q", offset) + uleb(len(chunk)) + chunk)
                    result = icp.run(["canister", "call", "decision-engine", "upload_chunk", "--args-file", str(argsfile), "--args-format", "bin", "-e", icp.env, "--candid", icp.did["decision-engine"]], timeout=300)
                    if "Ok" not in result:
                        raise Failure(result)
                    offset += len(chunk)
                    if offset % (64 * 1024 * 1024) == 0:
                        print(f"uploaded {offset // (1024*1024)} MiB", flush=True)
    warmup(icp, len(manifest["tensors"]))
    print(f"ready: {manifest['format']}, {offset} uploaded bytes", flush=True)


def warmup(icp, count=None):
    result = icp.call("decision-engine", "start_warmup")
    if "Ok" not in result:
        raise Failure(result)
    for i in range(count or 1024):
        result = icp.call("decision-engine", "warmup_next", timeout=600)
        if "Ok" not in result:
            raise Failure(result)
        if "true" in result:
            if count is not None and i != count-1:
                raise Failure("unexpected warmup tensor count")
            return
        if (i+1) % 32 == 0:
            print(f"loaded {i+1} tensors", flush=True)
    raise Failure("warmup did not complete")


def parse_costs(reply):
    start=reply.index("costs = vec {")+len("costs = vec {")
    depth=1;end=start
    while depth:
        if reply[end]=="{": depth+=1
        elif reply[end]=="}": depth-=1
        end+=1
    body=reply[start:end-1];records=[];pos=0
    while True:
        begin=body.find("record {",pos)
        if begin<0:break
        begin+=len("record {");end=begin;depth=1
        while depth:
            if body[end]=="{":depth+=1
            elif body[end]=="}":depth-=1
            end+=1
        record=body[begin:end-1];pos=end
        name=re.search(r'name\s*=\s*"([^"]+)"',record)[1]
        shape=re.search(r'shape\s*=\s*vec\s*\{([^}]*)\}',record)[1]
        shape=[int(v.strip().split(":")[0].replace("_","")) for v in shape.split(";") if v.strip()]
        instructions=int(re.search(r'instructions\s*=\s*([\d_]+)',record)[1].replace("_",""))
        records.append(dict(name=name,shape=shape,instructions=instructions))
    return records


def infer(icp, input_path, stepped=False, profile=False):
    inp = json.loads(input_path.read_text())
    ids = "; ".join(str(v) for v in inp["input_ids"])
    markers = "; ".join(str(v) for v in inp["markers"])
    args = f"(record {{ input_ids = vec {{ {ids} }}; markers = vec {{ {markers} }}; qtype_id = {inp['qtype_id']} : nat32 }})"
    started = time.monotonic()
    costs = []
    profiles = []
    if stepped:
        result = icp.call("decision-engine", "start_token_inference", args)
        if "Ok" not in result:
            raise Failure(result)
        ids = decode_blobs(result)
        if len(ids) != 1 or len(ids[0]) != 32:
            raise Failure(f"invalid job id: {result}")
        job = blob(ids[0])
        total = int(re.search(r"total\s*=\s*([\d_]+)", result)[1].replace("_", ""))
        for step in range(total):
            result = icp.call("decision-engine", "profile_token_step" if profile else "step_token_inference", f"({job}, {step} : nat32)", timeout=600)
            if profile and "Ok" in result:
                profiles.append(parse_costs(result))
                result=result[result.index("progress = record"):]
                result="Ok " + result
            if "Ok" not in result:
                raise Failure(result)
            costs.append(int(re.search(r"(?<!_)instructions\s*=\s*([\d_]+)", result)[1].replace("_", "")))
            print(f"step {step+1}/{total}: {costs[-1]} instructions", flush=True)
    else:
        result = icp.call("decision-engine", "infer_tokens", args, timeout=600)
    if "Ok" not in result:
        raise Failure(result)
    match = re.search(r"logits\s*=\s*(?:opt\s+)?vec\s*\{([^}]*)\}", result)
    instructions = re.search(r"cumulative_instructions\s*=\s*([\d_]+)" if stepped else r"instructions\s*=\s*([\d_]+)", result)
    if not match or not instructions:
        raise Failure(f"unexpected inference result: {result}")
    logits = [float(v.strip().split(":")[0].replace("_", "")) for v in match[1].split(";") if v.strip()]
    return {"input_tokens": len(inp["input_ids"]), "logits": logits,
            "instructions": int(instructions[1].replace("_", "")), "step_instructions": costs, "profiles": profiles, "local_wall_seconds": time.monotonic()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, help="upload and warm this pack before inference")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--profile", action="store_true", help="record detailed spans for each step")
    parser.add_argument("--stepped", action="store_true", help="one layer per update, for the full-size model")
    parser.add_argument("--warmup", action="store_true", help="rebuild model from already uploaded stable bytes")
    parser.add_argument("--initialize", action="store_true", help="create and install a fresh local canister")
    parser.add_argument("--identity", default="ic-laya-int8")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    icp = Icp(ROOT, "local", args.identity)
    require_local_network(icp)
    if network_status(icp) is None:
        raise Failure("start this project's local network first")
    if args.initialize:
        owner = ensure_identity(icp, args.identity)
        ensure_cycles(icp)
        ensure_canister(icp, "decision-engine")
        icp.run(["canister", "install", "decision-engine", "-e", "local", "-y", "-m", "install", "--wasm", str(BUILD / "decision-engine.wasm"), "--args", f"({principal(owner)})"], timeout=600)
    if args.pack:
        upload(icp, args.pack)
    elif args.warmup:
        warmup(icp)
    result = infer(icp, args.input, args.stepped or args.profile, args.profile)
    status = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
    result.update(network="local", measured_at=datetime.now(timezone.utc).isoformat(),
                  canister=status["id"], module_hash=status["module_hash"],
                  total_canister_memory_bytes=int(status["memory_size"].replace("_", "")),
                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest())
    active = decode_blobs(icp.query("decision-engine", "info"))
    if len(active) != 1 or len(active[0]) != 32:
        raise Failure("invalid active bundle response")
    result["bundle_sha256"] = active[0].hex()
    if args.pack and result["bundle_sha256"] != hashlib.sha256((args.pack / "manifest.json").read_bytes()).hexdigest():
        raise Failure("active bundle does not match uploaded manifest")
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
