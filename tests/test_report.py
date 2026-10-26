import json
import pandas as pd
from xai_industrial.reporting.export import incident_json


def test_incident_report_is_valid_json():
    payload = json.loads(incident_json(
        {"model": "IsolationForest"},
        {"row_index": 3},
        pd.DataFrame([{"sensor": "pressure", "deviation_score": 2.1}]),
    ))
    assert payload["summary"]["model"] == "IsolationForest"
    assert payload["ranked_sensor_deviations"][0]["sensor"] == "pressure"
    assert payload["limitations"]
