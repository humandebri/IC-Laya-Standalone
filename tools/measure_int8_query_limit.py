#!/usr/bin/env python3
"""Probe token/marker/qtype costs for a conservative raw Laya query limit."""
import hashlib
import json
from pathlib import Path
import tempfile
from datetime import datetime, timezone

from canister_infer import Icp, ROOT, decode_blobs, infer, network_status, require_local_network


def input_for(tokens, marker_count, qtype):
    # Raw TokenInput accepts ordered MASK positions. Spread them to exercise the
    # marker-only decision layer at every supported output width.
    ids = [50281] + [42] * (tokens - 2) + [50282]
    positions = [1 + (i * (tokens - 2)) // marker_count for i in range(marker_count)]
    assert len(set(positions)) == marker_count
    for position in positions:
        ids[position] = 50284
    return {'input_ids': ids, 'markers': positions, 'qtype_id': qtype}


def main():
    icp = Icp(ROOT, 'local', 'ic-laya-int8')
    require_local_network(icp)
    if network_status(icp) is None:
        raise RuntimeError('local network is stopped')
    status = json.loads(icp.run(['canister', 'status', 'decision-engine', '-e', 'local', '--json']))
    expected = '0x' + hashlib.sha256((ROOT / 'build/decision-engine.wasm').read_bytes()).hexdigest()
    if status['module_hash'] != expected:
        raise RuntimeError('installed module differs from build artifact')
    bundle = decode_blobs(icp.query('decision-engine', 'info'))[0].hex()
    report = {'measured_at': datetime.now(timezone.utc).isoformat(), 'network': 'local',
              'module_hash': expected, 'bundle_sha256': bundle, 'rows': [],
              'note': 'Raw synthetic inputs exercise all supported marker counts and qtypes; they are not schema or quality examples.'}
    output = ROOT / 'artifacts/int8_optimization_v4/query-limit-grid.json'
    with tempfile.TemporaryDirectory(prefix='laya-query-limit-') as directory:
        path = Path(directory) / 'input.json'
        for tokens in (15, 16, 17):
            for marker_count in range(2, 8):
                for qtype in range(3):
                    inp = input_for(tokens, marker_count, qtype)
                    path.write_text(json.dumps(inp))
                    result = infer(icp, path)
                    row = {'tokens': tokens, 'markers': marker_count, 'qtype_id': qtype,
                           'input': inp, 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'instructions': result['instructions'], 'logits': result['logits']}
                    report['rows'].append(row)
                    output.write_text(json.dumps(report, indent=2) + '\n')
                    print(tokens, marker_count, qtype, result['instructions'], flush=True)
    ending = json.loads(icp.run(['canister', 'status', 'decision-engine', '-e', 'local', '--json']))
    if ending['module_hash'] != expected:
        raise RuntimeError('module changed during grid')
    report['module_unchanged'] = True
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
