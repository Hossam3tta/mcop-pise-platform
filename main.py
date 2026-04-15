from __future__ import annotations

import json

from mcop.data_collector import get_current_metrics
from mcop.ai_optimizer import analyze_metrics
from mcop.decision_engine import decide_action
from mcop.executor import execute_action
from pise.predictor import predict


def run_pipeline() -> dict:
    metrics = get_current_metrics()
    analysis = analyze_metrics(metrics)
    prediction = predict(metrics)
    decision = decide_action(metrics, analysis, prediction)

    result = {
        "metrics": metrics,
        "analysis": analysis,
        "prediction": prediction,
        "decision": decision,
    }

    execute_action(decision)
    return result


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2))