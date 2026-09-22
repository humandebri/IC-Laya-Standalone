#!/usr/bin/env python3
"""Check retry/error/authorization behavior after a completed local stepped job."""
import json
import re
from canister_infer import Icp, ROOT, blob, decode_blobs, require_local_network, network_status


def main():
    icp = Icp(ROOT, "local", "ic-laya-int8")
    require_local_network(icp)
    assert network_status(icp), "local network must be running"
    current = icp.query("decision-engine", "token_inference_status")
    job = decode_blobs(current)[0]
    completed = int(re.search(r"completed\s*=\s*([\d_]+)", current)[1].replace("_", ""))
    total = int(re.search(r"total\s*=\s*([\d_]+)", current)[1].replace("_", ""))
    assert completed == total and total > 0
    replay = icp.call("decision-engine", "step_token_inference", f"({blob(job)}, {total-1} : nat32)")
    assert replay == current, "retry must return the same result without another step"
    assert "Transition" in icp.call("decision-engine", "step_token_inference", f"({blob(job)}, {total} : nat32)")
    assert "BindingMismatch" in icp.call("decision-engine", "step_token_inference", f"({blob(job)}, 0 : nat32)")
    wrong = bytes([job[0] ^ 1]) + job[1:]
    assert "BindingMismatch" in icp.call("decision-engine", "step_token_inference", f"({blob(wrong)}, 0 : nat32)")
    args = '(record { input_ids = vec { 1; 2 }; markers = vec { 0; 1 }; qtype_id = 0 : nat32 })'
    assert "Err" in icp.call("decision-engine", "start_token_inference", args)
    assert icp.query("decision-engine", "token_inference_status") == current, "invalid input must preserve job"
    stranger = Icp(ROOT, "local", "anonymous")
    for method, arg in [("start_token_inference", args), ("step_token_inference", f"({blob(job)}, 0 : nat32)"), ("profile_token_step", f"({blob(job)}, 0 : nat32)"), ("infer_tokens", args)]:
        assert "Unauthorized" in stranger.call("decision-engine", method, arg)
    assert "Unauthorized" in stranger.query("decision-engine", "token_inference_status")
    report = {"retry_same_result": True, "finished_step_rejected": True,
              "stale_step_rejected": True, "wrong_job_rejected": True,
              "invalid_input_preserves_job": True, "unauthorized_rejected": True}
    (ROOT / "artifacts/int8_canister_protocol.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
