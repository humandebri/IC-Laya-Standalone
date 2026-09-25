#!/usr/bin/env python3
"""Measure genuine short Choice/Noul token sequences on the local Laya canister."""
import hashlib
import json
from pathlib import Path
import tempfile
from datetime import datetime, timezone

from tokenizers import Tokenizer

from canister_infer import Icp, ROOT, decode_blobs, infer, network_status, require_local_network


CASES = (
    ('choice-choose', 'choice question: Choose.', ('red', 'blue', 'green'), 'blue', 0),
    ('choice-color-question', 'choice question: Color?', ('red', 'blue', 'green'), 'blue', 0),
    ('choice-color', 'choice question: Pick a color.', ('red', 'blue', 'green'), 'blue', 0),
    ('noul-safe', 'noul question: Safe?', ('no', 'yes'), 'yes', 2),
    ('noul-is-safe', 'noul question: Is this safe?', ('no', 'yes'), 'yes', 2),
)


def main():
    manifest = json.loads((ROOT / 'checkpoints/laya-int8/manifest.json').read_text())
    raw = (ROOT / 'checkpoints/laya-int8/tokenizer.json').read_bytes()
    tokenizer_hash = hashlib.sha256(raw).digest()
    if list(tokenizer_hash) != manifest['tokenizer_sha256']:
        raise RuntimeError('tokenizer differs from the active pack')
    tok = Tokenizer.from_file(str(ROOT / 'checkpoints/laya-int8/tokenizer.json'))
    encode = lambda value: tok.encode(value, add_special_tokens=False).ids
    icp = Icp(ROOT, 'local', 'ic-laya-int8')
    require_local_network(icp)
    if network_status(icp) is None:
        raise RuntimeError('local network is stopped')
    status = json.loads(icp.run(['canister', 'status', 'decision-engine', '-e', 'local', '--json']))
    bundle_hash = decode_blobs(icp.query('decision-engine', 'info'))[0].hex()
    if bundle_hash != 'bb70b3f0f2806bef5d4b670f44bb606892067fc0ebd928bd682b98ebdb2dc092':
        raise RuntimeError('unexpected active pack')
    report = {'measured_at': datetime.now(timezone.utc).isoformat(), 'network': 'local',
              'module_hash': status['module_hash'], 'bundle_sha256': bundle_hash,
              'tokenizer_sha256': tokenizer_hash.hex(), 'cases': [],
              'note': 'Short valid-looking raw inputs, not a quality or registered-schema test.'}
    for name, question, options, state, qtype in CASES:
        ids = [50281, *encode(question), 50282]
        markers = []
        for option in options:
            markers.append(len(ids))
            ids.extend([50284, *encode(' ' + option)])
        ids.extend([50282, *encode(state), 50282])
        inp = {'input_ids': ids, 'markers': markers, 'qtype_id': qtype}
        with tempfile.TemporaryDirectory(prefix='laya-short-') as directory:
            path = Path(directory) / 'input.json'
            path.write_text(json.dumps(inp))
            result = infer(icp, path)
        report['cases'].append({'name': name, 'question': question, 'options': options,
                                'state': state, 'input': inp,
                                'input_sha256': hashlib.sha256(json.dumps(inp, sort_keys=True).encode()).hexdigest(),
                                'tokens': len(ids), 'result': result})
        print(name, len(ids), result['instructions'], flush=True)
    output = ROOT / 'artifacts/int8_optimization_v4/short-inputs.json'
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
