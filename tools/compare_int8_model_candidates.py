#!/usr/bin/env python3
"""Compare full Laya inference on an isolated icp-cli local network."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

LAYA = Path(__file__).resolve().parents[1]
from measure_inference import Icp
from canister_infer import decode_blobs, infer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', required=True)
    p.add_argument('--network-root', type=Path, required=True)
    p.add_argument('--expected-wasm', type=Path, required=True)
    p.add_argument('--corpus', type=Path, default=LAYA / 'artifacts/int8_optimization_v4/validation-corpus.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--limit', type=int, default=96)
    a = p.parse_args()
    icp = Icp(a.network_root, 'local', 'ic-laya-int8')
    status = json.loads(icp.run(['canister', 'status', 'decision-engine', '-e', 'local', '--json']))
    expected_hash = '0x' + hashlib.sha256(a.expected_wasm.read_bytes()).hexdigest()
    if status['module_hash'] != expected_hash:
        raise RuntimeError('installed module does not match --expected-wasm')
    corpus_path = a.corpus
    corpus = json.loads(corpus_path.read_text())['cases'][:a.limit]
    bundle = decode_blobs(icp.query('decision-engine', 'info'))[0].hex()
    report = {'variant': a.variant, 'module_hash': status['module_hash'],
              'bundle_sha256': bundle, 'wasm_sha256': expected_hash[2:], 'corpus_sha256': hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
              'measured_at': datetime.now(timezone.utc).isoformat(), 'cases': []}
    baseline = None
    if a.baseline:
        baseline_report = json.loads(a.baseline.read_text())
        if baseline_report['bundle_sha256'] != bundle:
            raise RuntimeError('bundle changed')
        baseline = {x['id']: x for x in baseline_report['cases']}
    if a.output.exists():
        old = json.loads(a.output.read_text())
        for key in ('module_hash', 'bundle_sha256', 'corpus_sha256'):
            if old[key] != report[key]:
                raise RuntimeError(f'{key} changed; cannot resume')
        report['cases'] = old['cases']
    done = {x['id'] for x in report['cases']}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='laya-full-model-') as tmp:
        case_file = Path(tmp) / 'case.json'
        for case in corpus:
            if case['id'] in done:
                continue
            case_file.write_text(json.dumps(case['input']))
            tokens = len(case['input']['input_ids'])
            measured = infer(icp, case_file, stepped=tokens > 100)
            row = {'id': case['id'], 'tokens': tokens, 'instructions': measured['instructions'],
                   'logits': measured['logits'], 'update_calls': measured['inference_update_calls']}
            if baseline:
                old = baseline[case['id']]
                row['baseline_instructions'] = old['instructions']
                row['instruction_reduction_percent'] = 100 * (1 - row['instructions'] / old['instructions'])
                row['max_abs_error'] = max(abs(x - y) for x, y in zip(old['logits'], row['logits']))
                row['decision_matches'] = max(range(len(row['logits'])), key=row['logits'].__getitem__) == max(range(len(old['logits'])), key=old['logits'].__getitem__)
            report['cases'].append(row)
            a.output.write_text(json.dumps(report, indent=2) + '\n')
            print(a.variant, len(report['cases']), case['id'], tokens, row['instructions'],
                  row.get('instruction_reduction_percent'), flush=True)
    if baseline:
        from statistics import median
        report['median_reduction_percent'] = median(x['instruction_reduction_percent'] for x in report['cases'])
        report['min_reduction_percent'] = min(x['instruction_reduction_percent'] for x in report['cases'])
        report['max_abs_error'] = max(x['max_abs_error'] for x in report['cases'])
        report['decision_disagreements'] = [x['id'] for x in report['cases'] if not x['decision_matches']]
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    if baseline and (report['max_abs_error'] != 0 or report['decision_disagreements']):
        raise RuntimeError('model output changed')
    print('complete', a.variant, len(report['cases']), report.get('median_reduction_percent'), flush=True)


if __name__ == '__main__':
    main()
