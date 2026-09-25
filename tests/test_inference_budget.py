"""Routing safeguards for the measured Laya W8A8 Wasm and pack."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from inference_budget import BUNDLE_HASH, MODULE_HASH, plan_inference  # noqa: E402


class InferenceBudgetTests(unittest.TestCase):
    def route(self, tokens, budget):
        return plan_inference(tokens, 3, 0, budget, MODULE_HASH, BUNDLE_HASH)

    def test_measured_practical_input_uses_one_update(self):
        plan = self.route(103, 35_000_000_000)
        self.assertEqual((plan.mode, plan.steps_per_call), ("direct", None))
        self.assertLessEqual(plan.estimated_direct_instructions, plan.effective_budget)

    def test_long_input_scales_split_width_with_budget(self):
        self.assertEqual((self.route(128, 35_000_000_000).mode,
                          self.route(128, 35_000_000_000).steps_per_call), ("stepped", 16))
        self.assertEqual(self.route(128, 20_000_000_000).steps_per_call, 8)
        self.assertEqual(self.route(128, 2_000_000_000).steps_per_call, 1)

    def test_calibration_refuses_unknown_model_and_too_small_budget(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            plan_inference(103, 3, 0, 35_000_000_000, "other", BUNDLE_HASH)
        with self.assertRaisesRegex(ValueError, "at least"):
            self.route(128, 1_999_999_999)


if __name__ == "__main__":
    unittest.main()
