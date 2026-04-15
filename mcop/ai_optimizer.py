from __future__ import annotations


def analyze_metrics(metrics: dict) -> dict:
    score = 0

    if metrics["cpu"] > 80:
        score += 3
    elif metrics["cpu"] > 65:
        score += 2

    if metrics["memory"] > 75:
        score += 2
    elif metrics["memory"] > 60:
        score += 1

    if metrics["network_latency"] > 80:
        score += 2
    elif metrics["network_latency"] > 50:
        score += 1

    if metrics["cost_per_hour"] > 3.5:
        score += 1

    if score >= 6:
        risk = "high"
    elif score >= 3:
        risk = "medium"
    else:
        risk = "low"

    optimization_target = "performance" if metrics["cpu"] > 75 else "cost"

    return {
        "score": score,
        "risk": risk,
        "optimization_target": optimization_target,
    }