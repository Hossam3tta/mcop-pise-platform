from __future__ import annotations


def system_response(cpu: float, memory: float, latency: float) -> float:
    return (0.6 * cpu) + (0.25 * memory) + (0.15 * (latency / 2))