#!/usr/bin/env python3
"""Measure per-tensor warmup instructions and observed canister memory locally."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path

from canister_infer import Icp, ROOT, decode_blobs, network_status, require_local_network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    icp = Icp(ROOT, 'local', 'ic-laya-int8')
    require_local_network(icp)
    if network_status(icp) is None:
        raise RuntimeError('local network is stopped')

    def status():
        value = json.loads(icp.run(['canister', 'status', 'decision-engine', '-e', 'local', '--json']))
        return value['module_hash'], int(value['memory_size'].replace('_', ''))

    module, memory_before = status()
    expected = '0x' + hashlib.sha256((ROOT / 'build/decision-engine.wasm').read_bytes()).hexdigest()
    if module != expected:
        raise RuntimeError('installed module does not match build artifact')
    bundle = decode_blobs(icp.query('decision-engine', 'info'))[0].hex()
    started = icp.call('decision-engine', 'start_warmup')
    match = re.search(r'Ok\s*=\s*([\d_]+)', started)
    if not match:
        raise RuntimeError(started)
    count = int(match[1].replace('_', ''))
    measurements = []
    memory_samples = []
    for index in range(count):
        reply = icp.call('decision-engine', 'warmup_next_profile', timeout=600)
        if 'Ok' not in reply:
            raise RuntimeError(reply)
        cost = re.search(r'instructions\s*=\s*([\d_]+)', reply)
        done = re.search(r'done\s*=\s*(true|false)', reply)
        if not cost or not done:
            raise RuntimeError(reply)
        measurements.append(int(cost[1].replace('_', '')))
        if bool(done[1] == 'true') != bool(index == count - 1):
            raise RuntimeError('unexpected warmup completion')
        if (index + 1) % 16 == 0 or index + 1 == count:
            current_module, memory = status()
            if current_module != module:
                raise RuntimeError('module changed during warmup')
            memory_samples.append({'loaded_tensors': index + 1, 'memory_bytes': memory})
            print(index + 1, sum(measurements), memory, flush=True)
    report = {'measured_at': datetime.now(timezone.utc).isoformat(),
              'network': 'local', 'module_hash': module, 'bundle_sha256': bundle,
              'tensor_count': count, 'warmup_instructions': sum(measurements),
              'per_tensor_instructions': measurements, 'memory_before_bytes': memory_before,
              'memory_samples': memory_samples,
              'observed_memory_peak_bytes': max([memory_before] + [x['memory_bytes'] for x in memory_samples]),
              'note': 'Status samples observe Wasm memory after calls, not transient native allocation peaks.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
