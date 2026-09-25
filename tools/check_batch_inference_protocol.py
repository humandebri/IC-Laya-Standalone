#!/usr/bin/env python3
"""Verify the completed 128-token, 16-step-batch job on the local test canister."""
import argparse
import json
from pathlib import Path
import re
from canister_infer import Icp, ROOT, blob, decode_blobs, require_local_network, network_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=ROOT / 'artifacts/int8_batched_choice-128.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/int8_batch_protocol.json')
    options = parser.parse_args()
    icp = Icp(ROOT, 'local', 'ic-laya-int8')
    require_local_network(icp)
    assert network_status(icp)
    current = icp.query('decision-engine', 'token_inference_status')
    job = decode_blobs(current)[0]
    number = lambda name: int(re.search(rf'{name}\s*=\s*([\d_]+)', current)[1].replace('_', ''))
    assert number('completed') == number('total') == 32
    args = lambda step, size: f'({blob(job)}, {step} : nat32, {size} : nat32)'
    assert icp.call('decision-engine', 'step_token_inference_batch', args(16,16)) == current
    for step, size, error in [(16,15,'BindingMismatch'), (0,16,'BindingMismatch'),
                               (32,16,'Transition'), (32,0,'Invalid'), (32,17,'Invalid')]:
        assert error in icp.call('decision-engine', 'step_token_inference_batch', args(step,size))
        assert icp.query('decision-engine', 'token_inference_status') == current
    wrong = bytes([job[0]^1]) + job[1:]
    assert 'BindingMismatch' in icp.call('decision-engine', 'step_token_inference_batch', f'({blob(wrong)}, 32 : nat32, 16 : nat32)')
    # A single-step call cannot replay an interior step of the previous batch.
    assert 'BindingMismatch' in icp.call('decision-engine', 'step_token_inference', f'({blob(job)}, 31 : nat32)')
    stranger = Icp(ROOT, 'local', 'anonymous')
    assert 'Unauthorized' in stranger.call('decision-engine', 'step_token_inference_batch', args(16,16))
    assert icp.query('decision-engine', 'token_inference_status') == current
    # Retry combined start after completion: return the original first-batch result,
    # preserving the final job. Reuse of the same request ID with changed input fails.
    evidence = json.loads(options.evidence.read_text())
    if evidence.get('start_request_id'):
        inp = json.loads((ROOT / 'artifacts/laya-choice-128-input.json').read_text())
        def start_args(value, limit=16):
            ids = '; '.join(map(str, value['input_ids']))
            markers = '; '.join(map(str, value['markers']))
            return f"(record {{ input_ids = vec {{ {ids} }}; markers = vec {{ {markers} }}; qtype_id = {value['qtype_id']} : nat32 }}, {blob(bytes.fromhex(evidence['start_request_id']))}, {limit} : nat32)"
        first = icp.call('decision-engine', 'start_token_inference_batch', start_args(inp))
        assert 'Ok' in first and decode_blobs(first)[0] == job
        assert int(re.search(r'completed\s*=\s*([\d_]+)', first)[1].replace('_','')) == 16
        changed = dict(inp, qtype_id=(inp['qtype_id']+1)%3)
        assert 'IdConflict' in icp.call('decision-engine', 'start_token_inference_batch', start_args(changed))
        assert 'IdConflict' in icp.call('decision-engine', 'start_token_inference_batch', start_args(inp, 8))
        assert 'Unauthorized' in stranger.call('decision-engine', 'start_token_inference_batch', start_args(inp))
        assert icp.query('decision-engine', 'token_inference_status') == current
    report = dict(exact_batch_retry=True, changed_limit_rejected=True, stale_start_rejected=True,
                  finished_rejected=True, invalid_batch_bounds_rejected=True, wrong_job_rejected=True,
                  single_step_cannot_replay_batch=True, unauthorized_rejected=True, state_unchanged=True)
    report['combined_start_retry_and_conflict'] = bool(evidence.get('start_request_id'))
    report['module_hash'] = json.loads(icp.run(['canister','status','decision-engine','-e','local','--json']))['module_hash']
    options.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
