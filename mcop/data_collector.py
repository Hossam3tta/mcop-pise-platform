from __future__ import annotations

import random
from datetime import datetime


def get_current_metrics() -> dict:
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "cpu": round(random.uniform(35, 95), 2),
        "memory": round(random.uniform(30, 90), 2),
        "network_latency": round(random.uniform(10, 120), 2),
        "cost_per_hour": round(random.uniform(0.8, 4.5), 2),
        "instances": random.randint(1, 4),
    }