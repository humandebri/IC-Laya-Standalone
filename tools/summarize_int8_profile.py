#!/usr/bin/env python3
"""Compare matching local-canister profiles; reject changed weights/inputs/logits."""
import argparse
from collections import Counter
import json
from pathlib import Path


def totals(record):
    result=Counter()
    for step in record["profiles"]:
        for span in step:
            result[span["name"]]+=span["instructions"]
    return result


def compare(before,after):
    for key in ["input_sha256","bundle_sha256","input_tokens","logits"]:
        if before[key]!=after[key]:
            raise ValueError(f"comparison changed {key}")
    a,b=totals(before),totals(after)
    if sum(a.values())>before["instructions"] or sum(b.values())>after["instructions"]:
        raise ValueError("overlapping spans: cannot sum inclusive nested measurements")
    return {"tokens":before["input_tokens"],"bit_equal_serialized_logits":True,
            "before_instructions":before["instructions"],"after_instructions":after["instructions"],
            "instruction_reduction_percent":100*(1-after["instructions"]/before["instructions"]),
            "instruction_speedup":before["instructions"]/after["instructions"],
            "before_local_wall_seconds":before["local_wall_seconds"],"after_local_wall_seconds":after["local_wall_seconds"],
            "spans":[{"name":name,"before":a[name],"after":b[name],"before_percent":100*a[name]/before["instructions"],"after_percent":100*b[name]/after["instructions"]} for name in sorted(a,key=a.get,reverse=True)],
            "unattributed_before":before["instructions"]-sum(a.values()),"unattributed_after":after["instructions"]-sum(b.values())}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("before",type=Path);p.add_argument("after",type=Path);p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();result=compare(json.loads(args.before.read_text()),json.loads(args.after.read_text()))
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
