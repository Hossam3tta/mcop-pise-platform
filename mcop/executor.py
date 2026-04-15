from __future__ import annotations

import json
from pathlib import Path


LOG_FILE = Path("data/actions_log.json")


def execute_action(action_result: dict) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    history = []
    if LOG_FILE.exists():
        try:
            history = json.loads(LOG_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []

    history.append(action_result)
    LOG_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"[EXECUTOR] Executing action: {action_result['action']}")
    print(f"[EXECUTOR] Reason: {action_result['reason']}")