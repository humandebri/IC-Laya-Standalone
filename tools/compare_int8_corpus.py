#!/usr/bin/env python3
"""Compare raw logits and chosen options against a fixed baseline corpus."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tolerance', type=float, default=0.002)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    if len(baseline) != 96 or len(candidate) != len(baseline):
        raise RuntimeError('expected 96 matching inputs')
    errors = []
    disagreements = []
    for old, new in zip(baseline, candidate):
        if old['id'] != new['id'] or len(old['logits']) != len(new['logits']):
            raise RuntimeError('candidate corpus order or output shape changed')
        errors.append({'id': old['id'], 'max_abs_error': max(abs(a - b) for a, b in zip(old['logits'], new['logits']))})
        if max(range(len(old['logits'])), key=old['logits'].__getitem__) != max(range(len(new['logits'])), key=new['logits'].__getitem__):
            disagreements.append(old['id'])
    worst = max(errors, key=lambda entry: entry['max_abs_error'])
    report = {'cases': len(errors), 'tolerance': args.tolerance, 'worst': worst,
              'decision_disagreements': disagreements,
              'passes': worst['max_abs_error'] <= args.tolerance and not disagreements,
              'errors': errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: value for k, value in report.items() if k != 'errors'}, indent=2))
    if not report['passes']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
