#!/usr/bin/env python3
"""Small, pre-labeled local-canister probe of practical Laya decisions.

This is a diagnostic sample, not a representative accuracy benchmark. It only
calls the already installed local canister and does not upload or replace its model.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

from tokenizers import Tokenizer, __version__ as tokenizers_version

from canister_infer import Icp, ROOT, decode_blobs, infer, network_status, require_local_network


# Labels were chosen before running the model. Options and labels use zero-based indexes.
CASES = (
    ("route-invoice", "choice", "Route this support request.", ("billing", "technical support", "cancellation"),
     "The invoice for my annual plan has the wrong company name and tax address. Please issue a corrected invoice.", 0),
    ("route-charge", "choice", "Route this support request.", ("billing", "technical support", "cancellation"),
     "My card was charged twice for one order. The two charges have the same date and amount. Please check the duplicate charge.", 0),
    ("route-crash", "choice", "Route this support request.", ("billing", "technical support", "cancellation"),
     "The app crashes every time I try to upload a photo from my tablet. I have already installed the latest version.", 1),
    ("route-login", "choice", "Route this support request.", ("billing", "technical support", "cancellation"),
     "I cannot sign in after changing phones. The password reset email never arrives, even though the address is correct.", 1),
    ("route-cancel", "choice", "Route this support request.", ("billing", "technical support", "cancellation"),
     "Please cancel my subscription before it renews next Monday. I do not want another month of service.", 2),
    ("route-renewal", "choice", "Route this support request.", ("billing", "technical support", "cancellation"),
     "I no longer use this product. Please turn off automatic renewal for my annual plan and confirm its end date.", 2),
    ("cancel-yes", "noul", "Does the customer request cancellation?", ("no", "yes"),
     "Please cancel my subscription today and send confirmation by email.", 1),
    ("cancel-no", "noul", "Does the customer request cancellation?", ("no", "yes"),
     "I do not want to cancel. I only need a copy of last month's invoice.", 0),
    ("cancel-renewal", "noul", "Does the customer request cancellation?", ("no", "yes"),
     "Please stop my plan from renewing next month. I will keep using it until the current term ends.", 1),
    ("cancel-upgrade", "noul", "Does the customer request cancellation?", ("no", "yes"),
     "I want to upgrade from the monthly plan to the annual plan. What will the new price be?", 0),
    ("route-long-billing", "choice", "Route this support request.", ("billing", "technical support", "cancellation"),
     "Our finance team is closing the books this afternoon. We paid for the annual plan yesterday and received access, but the invoice shows our old legal entity and an incorrect tax address. The account owner has already updated the company profile. We need a corrected invoice for the same payment so that finance can record it properly. There is no problem with sign-in or product access, and we do not want to stop the subscription.", 0),
    ("route-long-technical", "choice", "Route this support request.", ("billing", "technical support", "cancellation"),
     "I paid for the annual plan and can see the receipt, so payment itself is fine. Since the most recent app update, the export screen closes immediately when I select a report. I tried a second browser, signed out and back in, and confirmed that another team member sees the same crash. We need help restoring exports before tomorrow's meeting. Please do not change or cancel our subscription.", 1),
    ("security-low-own-login", "score", "How urgent is this security report?", ("low", "medium", "high"),
     "I signed in from my new phone and received the expected security alert. I recognize the device and location. No account settings have changed.", 0),
    ("security-low-approved-user", "score", "How urgent is this security report?", ("low", "medium", "high"),
     "Our administrator invited a new teammate yesterday. The login alert names that teammate, and the administrator confirms the login was approved. There are no unexpected account changes.", 0),
    ("security-medium-unknown-login", "score", "How urgent is this security report?", ("low", "medium", "high"),
     "A login from an unfamiliar city succeeded this morning. The account owner does not recognize the device. No settings were changed and no transfers have been recorded yet.", 1),
    ("security-high-transfer", "score", "How urgent is this security report?", ("low", "medium", "high"),
     "The account owner reports an unknown login from another country. The new session changed the payout address and completed a transfer. The owner says they did not approve either action.", 2),
)

QTYPE = {"choice": 0, "noul": 2, "score": 1}
SPECIAL = {"cls": 50281, "sep": 50282, "mask": 50284}


def make_input(tok, kind, question, options, state):
    encode = lambda text: tok.encode(text, add_special_tokens=False).ids
    ids = [SPECIAL["cls"], *encode(f"{kind} question: {question}"), SPECIAL["sep"]]
    markers = []
    for option in options:
        markers.append(len(ids))
        ids.extend([SPECIAL["mask"], *encode(" " + option)])
    ids.extend([SPECIAL["sep"], *encode(state), SPECIAL["sep"]])
    return {"input_ids": ids, "markers": markers, "qtype_id": QTYPE[kind]}


def main():
    pack = ROOT / "checkpoints/laya-int8"
    tokenizer_raw = (pack / "tokenizer.json").read_bytes()
    manifest = json.loads((pack / "manifest.json").read_text())
    tokenizer_hash = hashlib.sha256(tokenizer_raw).digest()
    if list(tokenizer_hash) != manifest["tokenizer_sha256"]:
        raise RuntimeError("tokenizer and pack do not match")
    tok = Tokenizer.from_file(str(pack / "tokenizer.json"))
    icp = Icp(ROOT, "local", "ic-laya-int8")
    require_local_network(icp)
    if network_status(icp) is None:
        raise RuntimeError("local network is stopped")
    status = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
    bundle_hash = decode_blobs(icp.query("decision-engine", "info"))[0].hex()
    expected_bundle = hashlib.sha256((pack / "manifest.json").read_bytes()).hexdigest()
    if bundle_hash != expected_bundle:
        raise RuntimeError("installed canister pack differs from local checkpoint")
    report = {"measured_at": datetime.now(timezone.utc).isoformat(),
              "mode": "single_update",
              "module_hash": status["module_hash"], "bundle_sha256": bundle_hash,
              "tokenizer_sha256": tokenizer_hash.hex(),
              "tokenizers_version": tokenizers_version, "cases": []}
    output = ROOT / "artifacts/int8_optimization_v4/practical-128-probe.json"
    for name, kind, question, options, state, expected in CASES:
        inp = make_input(tok, kind, question, options, state)
        if len(inp["input_ids"]) > 128:
            raise RuntimeError(f"{name} exceeds 128 tokens: {len(inp['input_ids'])}")
        with tempfile.TemporaryDirectory(prefix="laya-practical-") as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(inp))
            result = infer(icp, path)
        predicted = max(range(len(options)), key=result["logits"].__getitem__)
        report["cases"].append({"id": name, "kind": kind, "question": question,
                                "options": options, "state": state, "expected_index": expected,
                                "predicted_index": predicted, "correct": predicted == expected,
                                "input": inp, "result": result})
        report["correct"] = sum(case["correct"] for case in report["cases"])
        report["total"] = len(report["cases"])
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{name}: {len(inp['input_ids'])} tokens, expected={options[expected]}, "
              f"predicted={options[predicted]}, instructions={result['instructions']:,}", flush=True)
    ending = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
    if ending["module_hash"] != status["module_hash"]:
        raise RuntimeError("canister module changed during probe")
    print(f"{report['correct']}/{report['total']} correct; details: {output}")


if __name__ == "__main__":
    main()
