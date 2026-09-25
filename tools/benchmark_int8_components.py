#!/usr/bin/env python3
"""Compare integer-dot and F32-writeback costs on the local test canister."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from canister_infer import Icp, ROOT, decode_blobs, network_status, require_local_network

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/int8_optimization_v2/f32-components.json')
options = parser.parse_args()
icp = Icp(ROOT, 'local', 'ic-laya-int8')
require_local_network(icp)
assert network_status(icp)
status = json.loads(icp.run(['canister', 'status', 'decision-engine', '-e', 'local', '--json']))
module_hash = status['module_hash']
assert module_hash == '0x' + hashlib.sha256((ROOT / 'build/decision-engine.wasm').read_bytes()).hexdigest()
rows = []
for outputs, inputs in [(3072, 1024), (5248, 1024), (1024, 2624)]:
    full = icp.call('decision-engine', 'benchmark_int8_kernel', f'(128 : nat32, {outputs} : nat32, {inputs} : nat32)', timeout=600)
    split = icp.call('decision-engine', 'benchmark_int8_components', f'({outputs} : nat32, {inputs} : nat32)', timeout=600)
    assert 'Ok' in full and 'Ok' in split
    total = max(int(s.replace('_','')) for s in re.findall(r'\binstructions\s*=\s*([\d_]+)', full))
    get = lambda name: int(re.search(rf'\b{name}\s*=\s*([\d_]+)', split)[1].replace('_',''))
    diagnostic = {name: get(name) for name in ('total', 'integer_dots', 'f32_writeback')}
    checksum = decode_blobs(full)[0].hex()
    assert checksum == decode_blobs(split)[0].hex()
    assert diagnostic['integer_dots'] + diagnostic['f32_writeback'] <= diagnostic['total']
    row = dict(tokens=128, rows=outputs, cols=inputs, full_forward_instructions=total,
               diagnostic=diagnostic, checksum=checksum,
               dot_fraction_of_diagnostic=diagnostic['integer_dots']/diagnostic['total'],
               writeback_fraction_of_diagnostic=diagnostic['f32_writeback']/diagnostic['total'])
    rows.append(row)
    print(outputs, inputs, row['full_forward_instructions'], diagnostic, flush=True)
assert module_hash == json.loads(icp.run(['canister', 'status', 'decision-engine', '-e', 'local', '--json']))['module_hash']
report = dict(network='local', module_hash=module_hash, source_sha256=hashlib.sha256((ROOT/'crates/laya-candle/src/int8.rs').read_bytes()).hexdigest(), rows=rows,
              note='Per-tile counters perturb code generation. Diagnostic fractions are estimates, not an exact decomposition of the uninstrumented kernel.')
path = options.output
path.write_text(json.dumps(report, indent=2)+'\n')
