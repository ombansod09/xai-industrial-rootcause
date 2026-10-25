from __future__ import annotations
import json
from typing import Any
import pandas as pd


def incident_json(summary: dict[str, Any], selected_row: dict[str, Any], explanations: pd.DataFrame) -> str:
    payload = {
        "summary": summary,
        "selected_observation": selected_row,
        "ranked_sensor_deviations": explanations.to_dict(orient="records"),
        "limitations": [
            "Sensor rankings are candidate contributors, not proof of physical root cause.",
            "The anomaly score is not a calibrated probability.",
            "This prototype does not issue automatic machine-control or maintenance actions.",
        ],
    }
    return json.dumps(payload, indent=2, default=str, allow_nan=False)
