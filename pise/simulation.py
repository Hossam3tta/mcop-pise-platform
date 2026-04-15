from __future__ import annotations

from pise.system_model import system_response


def simulate_next_state(metrics: dict) -> dict:
    response = system_response(
        metrics["cpu"],
        metrics["memory"],
        metrics["network_latency"],
    )

    predicted_cpu = min(100, round(response * 1.08, 2))
    predicted_state = "overloaded" if predicted_cpu > 85 else "stable"

    return {
        "predicted_cpu": predicted_cpu,
        "predicted_state": predicted_state,
    }