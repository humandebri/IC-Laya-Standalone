#!/usr/bin/env python3
"""Compare short-input update/query results and record the actual query boundary."""
import hashlib
import json
from pathlib import Path
import re
import tempfile

from canister_infer import Icp, ROOT, decode_blobs, infer, network_status, require_local_network


def main():
    icp = Icp(ROOT, 'local', 'ic-laya-int8')
    require_local_network(icp)
    if network_status(icp) is None:
        raise RuntimeError('local network is stopped')
    status = json.loads(icp.run(['canister', 'status', 'decision-engine', '-e', 'local', '--json']))
    module_hash = '0x' + hashlib.sha256((ROOT / 'build/decision-engine.wasm').read_bytes()).hexdigest()
    if status['module_hash'] != module_hash:
        raise RuntimeError('installed module differs from build artifact')
    source = json.loads((ROOT / 'artifacts/int8_optimization_v4/short-inputs.json').read_text())
    bundle_hash = decode_blobs(icp.query('decision-engine', 'info'))[0].hex()
    if source['bundle_sha256'] != bundle_hash:
        raise RuntimeError('pack changed since short-input generation')
    rows = []
    with tempfile.TemporaryDirectory(prefix='laya-short-query-') as directory:
        path = Path(directory) / 'input.json'
        for case in source['cases']:
            inp = case['input']
            path.write_text(json.dumps(inp))
            update = infer(icp, path)
            ids = '; '.join(map(str, inp['input_ids']))
            markers = '; '.join(map(str, inp['markers']))
            args = f"(record {{ input_ids = vec {{ {ids} }}; markers = vec {{ {markers} }}; qtype_id = {inp['qtype_id']} : nat32 }})"
            reply = icp.run(['canister', 'call', 'decision-engine', 'infer_tokens_query', args,
                             '-e', 'local', '--candid', icp.did['decision-engine'], '--query'],
                            expect_ok=False)
            row = {'name': case['name'], 'tokens': case['tokens'], 'input_sha256': case['input_sha256'],
                   'update_instructions': update['instructions'], 'update_logits': update['logits']}
            if 'Ok = record' in reply:
                match = re.search(r'logits\s*=\s*vec\s*\{([^}]*)\}', reply)
                cost = re.search(r'instructions\s*=\s*([\d_]+)', reply)
                if not match or not cost:
                    raise RuntimeError(reply)
                logits = [float(x) for x in re.findall(r'([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*:\s*float32', match[1], re.I)]
                row.update(query_status='ok', query_instructions=int(cost[1].replace('_', '')),
                           query_logits=logits, logits_match=logits == update['logits'])
                if not row['logits_match'] or row['query_instructions'] >= 5_000_000_000:
                    raise RuntimeError(f'query result mismatch: {case["name"]}')
            elif 'TooLong' in reply:
                if case['tokens'] <= 16:
                    raise RuntimeError(f'valid query was rejected: {case["name"]}')
                row.update(query_status='too_long', error=reply.strip())
            else:
                raise RuntimeError(reply)
            rows.append(row)
            print(row['name'], row['tokens'], row['query_status'], row.get('query_instructions'), flush=True)
    stranger = Icp(ROOT, 'local', 'anonymous')
    first = source['cases'][0]['input']
    ids = '; '.join(map(str, first['input_ids']))
    markers = '; '.join(map(str, first['markers']))
    args = f"(record {{ input_ids = vec {{ {ids} }}; markers = vec {{ {markers} }}; qtype_id = {first['qtype_id']} : nat32 }})"
    anonymous = stranger.query('decision-engine', 'infer_tokens_query', args)
    if 'Unauthorized' not in anonymous:
        raise RuntimeError('anonymous query unexpectedly succeeded')
    end_status = json.loads(icp.run(['canister', 'status', 'decision-engine', '-e', 'local', '--json']))
    if end_status['module_hash'] != module_hash:
        raise RuntimeError('module changed during query benchmark')
    report = {'network': 'local', 'module_hash': module_hash, 'bundle_sha256': bundle_hash,
              'source_module_hash': source['module_hash'], 'query_instruction_limit': 5_000_000_000,
              'query_max_tokens': 16,
              'rows': rows, 'anonymous_rejected': True, 'module_unchanged': True,
              'note': 'Raw owner-only query, not certified and not a registered-schema or quality test.'}
    output = ROOT / 'artifacts/int8_optimization_v4/short-query-results.json'
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
