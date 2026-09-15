from xai_industrial.models.baseline import make_isolation_forest
import numpy as np
X = np.random.default_rng(42).normal(size=(500, 8))
m = make_isolation_forest(42)
m.fit(X)
print("smoke test OK:", m.predict(X[:10]).tolist())
