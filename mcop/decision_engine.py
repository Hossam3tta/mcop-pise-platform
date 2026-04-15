from __future__ import annotations


def decide_action(metrics: dict, analysis: dict, prediction: dict | None = None) -> dict:
    predicted_cpu = prediction["predicted_cpu"] if prediction else metrics["cpu"]

    if predicted_cpu > 85:
        return {
            "action": "scale_up",
            "reason": "Predicted CPU exceeds safe threshold",
            "priority": "high",
        }

    if metrics["cost_per_hour"] > 4.0 and metrics["cpu"] < 40:
        return {
            "action": "scale_down",
            "reason": "Cost is high while utilization is low",
            "priority": "medium",
        }

    if analysis["risk"] == "high" and metrics["network_latency"] > 90:
        return {
            "action": "optimize_network",
            "reason": "High risk combined with elevated latency",
            "priority": "high",
        }

    return {
        "action": "maintain",
        "reason": "System is operating within acceptable thresholds",
        "priority": "low",
    }