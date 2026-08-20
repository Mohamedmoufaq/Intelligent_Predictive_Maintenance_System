"""Business-facing risk, decision, and cost functions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CostAssumptions:
    """Example assumptions only; replace with actual organization costs when available."""

    preventive_maintenance_cost: float = 8_000
    potential_failure_cost: float = 120_000
    unnecessary_maintenance_cost: float = 2_000


def calculate_risk(probability: float) -> tuple[float, str]:
    """Convert failure probability to a presentation-friendly 0-100 risk score."""
    score = round(max(0.0, min(1.0, probability)) * 100, 1)
    if score < 25:
        level = "LOW"
    elif score < 50:
        level = "MEDIUM"
    elif score < 75:
        level = "HIGH"
    else:
        level = "CRITICAL"
    return score, level


def maintenance_decision(probability: float, threshold: float, risk_level: str) -> str:
    """Apply explicit project thresholds rather than treating a model label as a decision."""
    if probability >= threshold or risk_level == "CRITICAL":
        return "Immediate Maintenance"
    return {
        "HIGH": "Prepare Maintenance",
        "MEDIUM": "Schedule Maintenance",
        "LOW": "Continue Monitoring",
    }[risk_level]


def estimate_costs(probability: float, assumptions: CostAssumptions = CostAssumptions()) -> dict[str, float | str]:
    """Compare expected failure loss with an example preventive-maintenance cost."""
    expected_failure_loss = probability * assumptions.potential_failure_cost
    recommendation = (
        "Maintenance is financially justified"
        if expected_failure_loss > assumptions.preventive_maintenance_cost
        else "Continue monitoring; expected loss is below preventive maintenance cost"
    )
    return {
        "potential_failure_cost": assumptions.potential_failure_cost,
        "expected_failure_loss": round(expected_failure_loss, 2),
        "preventive_maintenance_cost": assumptions.preventive_maintenance_cost,
        "business_decision": recommendation,
    }
