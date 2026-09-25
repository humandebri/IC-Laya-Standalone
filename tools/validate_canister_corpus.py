#!/usr/bin/env python3
"""Run the fixed 96-input compatibility corpus on the installed local canister."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

from canister_infer import Icp, ROOT, decode_blobs, infer, network_status, require_local_network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=ROOT / 'artifacts/int8_optimization_v4/validation-corpus.json')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'artifacts/int8_optimization_v4/baseline-corpus-logits.json')
    parser.add_argument('--collect-only', action='store_true', help='record canister outputs without comparing to a baseline')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-module-hash', help='verified installed hash when build/ has moved to another candidate')
    args = parser.parse_args()
    icp = Icp(ROOT, 'local', 'ic-laya-int8')
    require_local_network(icp)
    if network_status(icp) is None:
        raise RuntimeError('local network is stopped')
    status = json.loads(icp.run(['canister', 'status', 'decision-engine', '-e', 'local', '--json']))
    expected_module = args.expected_module_hash or '0x' + hashlib.sha256((ROOT / 'build/decision-engine.wasm').read_bytes()).hexdigest()
    if status['module_hash'] != expected_module:
        raise RuntimeError('installed module does not match build artifact')
    corpus = json.loads(args.corpus.read_text())['cases']
    if args.collect_only:
        baseline = None
        baseline_sha = None
    else:
        raw_baseline = json.loads(args.baseline.read_text())
        entries = raw_baseline if isinstance(raw_baseline, list) else raw_baseline['cases']
        baseline = {entry['id']: entry for entry in entries}
        baseline_sha = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    if len(corpus) != 96 or (baseline is not None and len(baseline) != 96):
        raise RuntimeError('expected 96 cases')
    report = {'measured_at': datetime.now(timezone.utc).isoformat(), 'network': 'local',
              'module_hash': status['module_hash'],
              'bundle_sha256': decode_blobs(icp.query('decision-engine', 'info'))[0].hex(),
              'corpus_sha256': hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
              'baseline_sha256': baseline_sha,
              'cases': [], 'tolerance': 0.002}
    if args.output.exists():
        old = json.loads(args.output.read_text())
        for key in ('module_hash', 'bundle_sha256', 'corpus_sha256', 'baseline_sha256'):
            if old[key] != report[key]:
                raise RuntimeError(f'{key} changed; cannot resume')
        report['cases'] = old['cases']
    done = {case['id'] for case in report['cases']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='laya-corpus-') as temp:
        input_path = Path(temp) / 'input.json'
        for case in corpus:
            if case['id'] in done:
                continue
            input_path.write_text(json.dumps(case['input']))
            tokens = len(case['input']['input_ids'])
            measured = infer(icp, input_path, stepped=tokens > 100, steps_per_call=16)
            new = measured['logits']
            entry = {'id': case['id'], 'schema': case['schema'], 'kind': case['kind'],
                     'tokens': tokens, 'logits': new, 'instructions': measured['instructions'],
                     'update_calls': measured['inference_update_calls']}
            if baseline is not None:
                old_entry = baseline[case['id']]
                old = old_entry['logits']
                if len(old) != len(new):
                    raise RuntimeError(f"logit count changed: {case['id']}")
                entry['max_abs_error'] = max(abs(a - b) for a, b in zip(old, new))
                entry['decision_matches'] = (max(range(len(old)), key=old.__getitem__)
                                             == max(range(len(new)), key=new.__getitem__))
                if 'instructions' in old_entry:
                    entry['baseline_instructions'] = old_entry['instructions']
                    entry['instruction_change_percent'] = 100 * (measured['instructions'] / old_entry['instructions'] - 1)
            report['cases'].append(entry)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(len(report['cases']), case['id'], tokens, measured['instructions'], entry.get('max_abs_error'), flush=True)
    if baseline is None:
        report['passes'] = len(report['cases']) == 96
    else:
        report['worst_abs_error'] = max(case['max_abs_error'] for case in report['cases'])
        report['decision_disagreements'] = [case['id'] for case in report['cases'] if not case['decision_matches']]
        changes = [case['instruction_change_percent'] for case in report['cases'] if 'instruction_change_percent' in case]
        report['worst_instruction_regression_percent'] = max(changes) if changes else None
        report['median_instruction_change_percent'] = sorted(changes)[len(changes) // 2] if changes else None
        report['passes'] = (len(report['cases']) == 96 and report['worst_abs_error'] <= 0.002
                            and not report['decision_disagreements']
                            and (not changes or max(changes) <= 1.0))
    ending = json.loads(icp.run(['canister', 'status', 'decision-engine', '-e', 'local', '--json']))
    if ending['module_hash'] != report['module_hash']:
        raise RuntimeError('module changed during validation')
    report['module_unchanged'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not report['passes']:
        raise SystemExit('numerical validation failed')


if __name__ == '__main__':
    main()
