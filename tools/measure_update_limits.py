#!/usr/bin/env python3
"""Probe single-update limits with controlled token-length variants on local IC."""
import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from canister_infer import Icp, ROOT, Failure, infer, decode_blobs, require_local_network, network_status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lengths', default='80,81,82,83,84,85,86,87,88,89,90')
    p.add_argument('--cases', default='choice,noul,score')
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/int8_update_limits.json')
    args = p.parse_args()
    lengths = [int(n) for n in args.lengths.split(',')]
    assert all(1 <= n <= 128 for n in lengths)
    icp = Icp(ROOT, 'local', 'ic-laya-int8')
    require_local_network(icp)
    assert network_status(icp)
    status = json.loads(icp.run(['canister', 'status', 'decision-engine', '-e', 'local', '--json']))
    bundle = decode_blobs(icp.query('decision-engine', 'info'))[0].hex()
    report = dict(network='local', measured_at=datetime.now(timezone.utc).isoformat(),
                  canister=status['id'], module_hash=status['module_hash'], bundle_sha256=bundle,
                  note='Synthetic length variants: fixed prompt/options; repeat or truncate body tokens, preserve final SEP. Not a quality test or universal limit.', rows=[])
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        for key in ('canister', 'module_hash', 'bundle_sha256'):
            assert previous[key] == report[key], f'cannot resume after {key} changed'
        report['rows'] = previous['rows']
    seen = {(r['case'], r['input_tokens']) for r in report['rows']}
    with tempfile.TemporaryDirectory(prefix='laya-update-limit-') as tmp:
        for name in args.cases.split(','):
            source = ROOT / f'artifacts/laya-{name}-input.json'
            original = json.loads(source.read_text())
            ids = original['input_ids']
            # The first SEP after the last option marker ends the option prefix.
            end = ids.index(50282, original['markers'][-1]) + 1
            body = ids[end:-1]
            assert body and ids[-1] == 50282
            for length in lengths:
                if (name, length) in seen:
                    continue
                count = length - end - 1
                assert count > 0
                inp = dict(original, input_ids=ids[:end] + (body * ((count + len(body)-1)//len(body)))[:count] + [ids[-1]])
                path = Path(tmp) / 'input.json'
                path.write_text(json.dumps(inp))
                row = dict(case=name, input_tokens=length, input=inp,
                           input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                try:
                    row.update(infer(icp, path), status='ok')
                except Failure as e:
                    message = str(e)
                    if 'IC0522' not in message:
                        raise
                    row.update(status='instruction_limit', error=message)
                report['rows'].append(row)
                args.output.write_text(json.dumps(report, indent=2) + '\n')
                print(name, length, row['status'], row.get('instructions'), flush=True)
    final = json.loads(icp.run(['canister', 'status', 'decision-engine', '-e', 'local', '--json']))
    assert final['module_hash'] == report['module_hash']
    assert decode_blobs(icp.query('decision-engine', 'info'))[0].hex() == bundle
    report['module_and_bundle_unchanged'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
