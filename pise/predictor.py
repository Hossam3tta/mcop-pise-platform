from __future__ import annotations

from pise.simulation import simulate_next_state


def predict(metrics: dict) -> dict:
    return simulate_next_state(metrics)