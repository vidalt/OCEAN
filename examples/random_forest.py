import time

import gurobipy as gp
from datasets import load_credit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ocean import MIPExplainer
from ocean.typing import FloatArray1D

# Parameters
seed = 42
n_estimators = 100
max_depth = 5
n_examples = 2  # <= 5997
# Load the data
print("Loading the data")
mapper, (X, y) = load_credit()
print("Data loaded")

X_train, X_test, y_train, y_test = train_test_split(
    X.to_numpy(),
    y.to_numpy().flatten(),
    test_size=0.2,
    random_state=seed,
)

# Fit the Random Forest model
print("Fitting a Random Forest model")
rf = RandomForestClassifier(
    n_estimators=n_estimators, random_state=seed, max_depth=max_depth
)
rf.fit(X_train, y_train)
print("Model fitted")

# Evaluate the model
print("Building the MIPExplainer")
env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.setParam("Seed", seed)
env.start()
start = time.time()
mip = MIPExplainer(rf, mapper=mapper, env=env)
end = time.time()
print("MIPExplainer built")

elapsed = end - start

# Generate multiple queries:
queries: list[tuple[FloatArray1D, int]] = []

y_pred = rf.predict(X_test)

print("Generating queries")
for x, y_ in zip(X_test[:n_examples], y_pred[:n_examples], strict=True):
    queries.append((x, 1 - y_))
print("Queries generated")

print("Running queries")
times: list[float] = []
for i, (x, class_) in enumerate(queries):
    start = time.time()
    mip.add_objective(x)
    mip.set_majority_class(class_)
    mip.optimize()
    mip.clear_majority_class()
    mip.cleanup()
    end = time.time()
    print(f"Query {i} completed in {end - start:.2f} seconds")

    times.append(end - start)
print("Queries run")

print(f"Building the model took {elapsed:.2f} seconds")
print(f"Average time per query: {sum(times) / len(times):.2f} seconds")
print(f"Maximum time per query: {max(times):.2f} seconds")
print(f"Minimum time per query: {min(times):.2f} seconds")
print(f"Total number of queries: {len(queries)}")
