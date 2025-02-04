import time
from pathlib import Path

import gurobipy as gp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ocean import MIPExplainer
from ocean.feature import Feature, FeatureMapper
from ocean.typing import FloatArray1D

# Parameters
seed = 42
n_estimators = 100
max_depth = 5
n_examples = 2  # <= 5997

file = Path(__file__).parent / "data" / "default_credit_numerical.csv"
target = "DEFAULT_PAYEMENT"

print(f"Reading data from {file}")
data = pd.read_csv(file)
X, y = (
    data.drop(columns=target).to_numpy(),
    data[target].astype(int).to_numpy(),
)
print("Data loaded")

features: list[Feature] = []
names: list[str] = []
columns: "pd.Index[str] | pd.MultiIndex" = pd.Index([""])

names.extend([
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "Sex",
    "Education",
    "Marriage",
    "Pay_0",
    "Pay_2",
    "Pay_3",
    "Pay_4",
    "Pay_5",
    "Pay_6",
])
features.extend([
    Feature(ftype=Feature.Type.CONTINUOUS, levels=(10000 - 1, 1000000 + 1)),
    Feature(ftype=Feature.Type.DISCRETE, levels=range(21, 80)),
    Feature(ftype=Feature.Type.CONTINUOUS, levels=(-165580 - 1, 964511 + 1)),
    Feature(ftype=Feature.Type.CONTINUOUS, levels=(-69777 - 1, 983931 + 1)),
    Feature(ftype=Feature.Type.CONTINUOUS, levels=(-157264 - 1, 1664089 + 1)),
    Feature(ftype=Feature.Type.CONTINUOUS, levels=(-170000 - 1, 891586 + 1)),
    Feature(ftype=Feature.Type.CONTINUOUS, levels=(-81334 - 1, 927171 + 1)),
    Feature(ftype=Feature.Type.CONTINUOUS, levels=(-339603 - 1, 961664 + 1)),
    Feature(ftype=Feature.Type.CONTINUOUS, levels=(0 - 1, 873552 + 1)),
    Feature(ftype=Feature.Type.CONTINUOUS, levels=(0 - 1, 1684259 + 1)),
    Feature(ftype=Feature.Type.CONTINUOUS, levels=(0 - 1, 896040 + 1)),
    Feature(ftype=Feature.Type.CONTINUOUS, levels=(0 - 1, 621000 + 1)),
    Feature(ftype=Feature.Type.CONTINUOUS, levels=(0 - 1, 426529 + 1)),
    Feature(ftype=Feature.Type.CONTINUOUS, levels=(0 - 1, 528666 + 1)),
    Feature(ftype=Feature.Type.BINARY),
    Feature(
        ftype=Feature.Type.ONE_HOT_ENCODED,
        codes=("Graduate_School", "High_School", "Others", "University"),
    ),
    Feature(
        ftype=Feature.Type.ONE_HOT_ENCODED,
        codes=("Married", "Others", "Single"),
    ),
    Feature(ftype=Feature.Type.BINARY),
    Feature(ftype=Feature.Type.BINARY),
    Feature(ftype=Feature.Type.BINARY),
    Feature(ftype=Feature.Type.BINARY),
    Feature(ftype=Feature.Type.BINARY),
    Feature(ftype=Feature.Type.BINARY),
])

columns = pd.MultiIndex.from_tuples([
    ("LIMIT_BAL", ""),
    ("AGE", ""),
    ("BILL_AMT1", ""),
    ("BILL_AMT2", ""),
    ("BILL_AMT3", ""),
    ("BILL_AMT4", ""),
    ("BILL_AMT5", ""),
    ("BILL_AMT6", ""),
    ("PAY_AMT1", ""),
    ("PAY_AMT2", ""),
    ("PAY_AMT3", ""),
    ("PAY_AMT4", ""),
    ("PAY_AMT5", ""),
    ("PAY_AMT6", ""),
    ("Sex", ""),
    ("Education", "Graduate_School"),
    ("Education", "High_School"),
    ("Education", "Others"),
    ("Education", "University"),
    ("Marriage", "Married"),
    ("Marriage", "Others"),
    ("Marriage", "Single"),
    ("Pay_0", ""),
    ("Pay_2", ""),
    ("Pay_3", ""),
    ("Pay_4", ""),
    ("Pay_5", ""),
    ("Pay_6", ""),
])

mapper = FeatureMapper(names=names, features=features, columns=columns)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
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
