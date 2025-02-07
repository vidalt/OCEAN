import gurobipy as gp
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ocean.datasets import load_adult
from ocean.explainer import MIPExplainer

mapper, (data, target) = load_adult()

rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
rf.fit(data, target)

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

model = MIPExplainer(rf, mapper=mapper, env=env)

generator = np.random.default_rng(42)
i = generator.integers(0, data.shape[0])
x = data.iloc[[i]]
y = int(rf.predict(x)[0])

model.set_majority_class(m_class=1 - y)
model.optimize()

print(model.solution)
