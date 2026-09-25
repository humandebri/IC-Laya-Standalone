"""Empirical, hash-bound update routing for the current Laya W8A8 canister.

This estimates cost without running the model. It is a routing hint, not an IC
instruction limit or a mathematically proven upper bound.
"""
from dataclasses import dataclass


MODULE_HASH = "0xdd7013df97f0b2b039540cb94888aebd7239efe085d53935601f6206236d1b49"
BUNDLE_HASH = "bb70b3f0f2806bef5d4b670f44bb606892067fc0ebd928bd682b98ebdb2dc092"
MAX_SAFE_DIRECT_BUDGET = 35_000_000_000
MIN_BUDGET = 2_000_000_000

# Above every observed one-shot cost in the 96-input compatibility corpus and
# 16 practical examples, including the measured 128-token cases. Recalibrate
# when either hash changes. The smallest observed headroom is about 1.24B.
DIRECT_BASE = 1_200_000_000
DIRECT_PER_TOKEN = 310_000_000

# Conservative rounded thresholds from the current 128-token, 32-step profile.
# The measured maximum for each consecutive width was 1.358B, 2.716B, 5.431B,
# 10.845B, and 21.665B respectively. These thresholds leave >=25% headroom.
STEP_THRESHOLDS = ((16, 28_000_000_000), (8, 14_000_000_000),
                   (4, 7_000_000_000), (2, 4_000_000_000),
                   (1, MIN_BUDGET))


@dataclass(frozen=True)
class InferencePlan:
    mode: str
    steps_per_call: int | None
    estimated_direct_instructions: int
    requested_budget: int
    effective_budget: int


def plan_inference(input_tokens: int, marker_count: int, qtype_id: int,
                   requested_budget: int, module_hash: str,
                   bundle_hash: str) -> InferencePlan:
    if module_hash != MODULE_HASH or bundle_hash != BUNDLE_HASH:
        raise ValueError("instruction calibration does not match installed Wasm and pack")
    if not 1 <= input_tokens <= 128 or not 2 <= marker_count <= 7 or qtype_id not in (0, 1, 2):
        raise ValueError("input is outside the calibrated model shape")
    if requested_budget < MIN_BUDGET:
        raise ValueError(f"budget must be at least {MIN_BUDGET:,} instructions")
    effective_budget = min(requested_budget, MAX_SAFE_DIRECT_BUDGET)
    direct = DIRECT_BASE + DIRECT_PER_TOKEN * input_tokens
    if direct <= effective_budget:
        return InferencePlan("direct", None, direct, requested_budget, effective_budget)
    for steps, threshold in STEP_THRESHOLDS:
        if threshold <= effective_budget:
            return InferencePlan("stepped", steps, direct, requested_budget, effective_budget)
    raise AssertionError("minimum budget has no split route")
