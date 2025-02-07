import gurobipy as gp
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ocean.datasets import load_adult
from ocean.mip import FeatureVar, Model
from ocean.tree import parse_trees


def get_value(var: FeatureVar) -> float | str | int | None:
    if not var.is_one_hot_encoded:
        if var.is_binary:
            return int(var.X)

        if np.isclose(var.X, 0.0):
            return 0.0
        return var.X
    for code in var.codes:
        if var[code].X == 1:
            return code
    msg = "No code has been selected."
    raise ValueError(msg)


def print_dict[K, V](dictionary: dict[K, V]) -> None:
    max_key_length = max(len(set(k)) for k in dictionary)
    for k, v in dictionary.items():
        print(f"{str(k).ljust(max_key_length + 1)} :\t{v}")


mapper, (data, target) = load_adult()

x, y = data.to_numpy(), target.to_numpy()

rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
rf.fit(x, y)

trees = parse_trees(rf, mapper=mapper)

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

model = Model(trees, mapper, env=env)
model.build()

generator = np.random.default_rng(42)
i = generator.integers(0, x.shape[0])
x_ = x[i]
y_ = rf.predict(x_.reshape(1, -1))[0]

model.set_majority_class(m_class=1 - y_)
model.optimize()

new_x = dict(model.solution.reduce(get_value))
print_dict(new_x)
