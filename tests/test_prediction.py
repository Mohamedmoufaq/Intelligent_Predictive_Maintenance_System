"""Smoke tests for the saved predictive-maintenance inference pipeline."""

import unittest

from Src.predict import load_artifacts, predict_machine


class PredictionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = load_artifacts()
        cls.sample = {
            "Type": "M",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 0,
        }

    def test_prediction_returns_business_fields(self):
        result = predict_machine(self.sample, self.artifacts)
        self.assertIsInstance(result["machine_failure"], bool)
        self.assertGreaterEqual(result["failure_probability"], 0.0)
        self.assertLessEqual(result["failure_probability"], 1.0)
        self.assertIn(result["risk_level"], {"LOW", "MEDIUM", "HIGH", "CRITICAL"})
        self.assertIn("maintenance_recommendation", result)
        self.assertIn("expected_failure_loss", result)

    def test_missing_feature_is_rejected(self):
        incomplete = self.sample.copy()
        incomplete.pop("Torque [Nm]")
        with self.assertRaises(ValueError):
            predict_machine(incomplete, self.artifacts)


if __name__ == "__main__":
    unittest.main()
