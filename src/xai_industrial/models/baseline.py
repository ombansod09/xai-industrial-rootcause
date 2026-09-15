from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

def make_isolation_forest(seed=42, contamination="auto"):
    return Pipeline([
        ("scaler", RobustScaler()),
        ("model", IsolationForest(
            n_estimators=300, contamination=contamination,
            random_state=seed, n_jobs=-1
        )),
    ])
