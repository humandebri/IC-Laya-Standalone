#!/usr/bin/env python3
"""Repeat the installed canister's bounded INT8 kernel benchmark.

The benchmark constructs deterministic synthetic tensors inside the canister.
Instruction counts exclude tensor construction and output checksumming; the
reported checksum still detects changes to the kernel's numerical results.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics

from canister_infer import Icp, ROOT, decode_blobs, network_status, require_local_network


SHAPES = ((3072, 1024), (5248, 1024), (1024, 2624))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', default='28,38,64,96,128')
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lengths = [int(value) for value in args.tokens.split(',')]
    if not lengths or any(not 1 <= value <= 128 for value in lengths):
        parser.error('tokens must be in 1..=128')
    if args.warmups < 0 or args.repeats < 1:
        parser.error('warmups must be nonnegative and repeats positive')

    icp = Icp(ROOT, 'local', 'ic-laya-int8')
    require_local_network(icp)
    if network_status(icp) is None:
        raise RuntimeError('local network is stopped')
    status = json.loads(icp.run(['canister', 'status', 'decision-engine', '-e', 'local', '--json']))
    wasm_hash = hashlib.sha256((ROOT / 'build/decision-engine.wasm').read_bytes()).hexdigest()
    if status['module_hash'] != '0x' + wasm_hash:
        raise RuntimeError('installed canister does not match build/decision-engine.wasm')
    active_bundle = decode_blobs(icp.query('decision-engine', 'info'))[0].hex()
    report = {
        'measured_at': datetime.now(timezone.utc).isoformat(),
        'network': 'local',
        'canister': status['id'],
        'module_hash': status['module_hash'],
        'bundle_sha256': active_bundle,
        'kernel_source_sha256': hashlib.sha256((ROOT / 'crates/laya-candle/src/int8.rs').read_bytes()).hexdigest(),
        'warmups': args.warmups,
        'repeats': args.repeats,
        'rows': [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for tokens in lengths:
        for rows, cols in SHAPES:
            samples = []
            checksums = set()
            for iteration in range(args.warmups + args.repeats):
                reply = icp.call('decision-engine', 'benchmark_int8_kernel',
                                 f'({tokens} : nat32, {rows} : nat32, {cols} : nat32)', timeout=600)
                if 'Ok' not in reply:
                    raise RuntimeError(reply)
                count = max(int(value.replace('_', '')) for value in
                            re.findall(r'\binstructions\s*=\s*([\d_]+)', reply))
                checksums.add(decode_blobs(reply)[0].hex())
                if iteration >= args.warmups:
                    samples.append(count)
            if len(checksums) != 1:
                raise RuntimeError(f'checksum changed at {tokens}, {rows}, {cols}')
            entry = {'tokens': tokens, 'rows': rows, 'cols': cols,
                     'instructions': samples, 'median': int(statistics.median(samples)),
                     'min': min(samples), 'max': max(samples), 'checksum': checksums.pop()}
            report['rows'].append(entry)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(tokens, rows, cols, entry['median'], flush=True)
    ending = json.loads(icp.run(['canister', 'status', 'decision-engine', '-e', 'local', '--json']))
    if ending['module_hash'] != report['module_hash']:
        raise RuntimeError('module changed during benchmark')
    report['module_unchanged'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
