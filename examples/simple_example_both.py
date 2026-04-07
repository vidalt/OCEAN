import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ocean import ConstraintProgrammingExplainer, MixedIntegerProgramExplainer
from ocean.datasets import load_adult

print_paths = True
plot_anytime_distances = True
num_workers = 8
random_state = 0
timeout = 3600


def to_frame(
    x: np.ndarray[tuple[int], np.dtype[np.float64]],
    columns: pd.Index | pd.MultiIndex,
) -> pd.DataFrame:
    return pd.DataFrame([x], columns=columns)


def print_close_threshold_paths(
    model: RandomForestClassifier | AdaBoostClassifier,
    cf: np.ndarray[tuple[int], np.dtype[np.float64]] | None,
    *,
    columns: pd.Index | pd.MultiIndex,
    query_pred: int,
    label: str,
) -> None:
    if cf is None:
        print(f"{label}: No CF found.")
        return

    cf_frame = to_frame(cf, columns)
    if int(model.predict(cf_frame)[0]) != query_pred:
        print(f"{label} Valid CF.")
        return

    print(f"INVALID {label} CF: decision path of the CF found by {label}")
    for i, estimator in enumerate(model.estimators_):
        if int(estimator.predict(cf_frame)[0]) != query_pred:
            continue

        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        node_indicator = estimator.decision_path(cf_frame)
        leaf_id = estimator.apply(cf_frame)
        sample_id = 0
        start = node_indicator.indptr[sample_id]
        stop = node_indicator.indptr[sample_id + 1]
        node_index = node_indicator.indices[start:stop]

        print(node_index)
        print(
            f"[Tree {i}] Rules used to predict sample {sample_id} "
            "with features close to a threshold:\n"
        )
        for node_id in node_index:
            if leaf_id[sample_id] == node_id:
                continue

            threshold_sign = (
                "<=" if cf[feature[node_id]] <= threshold[node_id] else ">"
            )
            if np.abs(cf[feature[node_id]] - threshold[node_id]) < 1e-3:
                print(
                    f"decision node {node_id}: "
                    f"cf[{feature[node_id]}] = {cf[feature[node_id]]} "
                    f"{threshold_sign} {threshold[node_id]}"
                )


def unpack_anytime(
    anytime: list[dict[str, float]] | None,
) -> tuple[list[float], list[float]]:
    if anytime is None:
        return [], []
    objectives = [entry.get("objective_value", 0.0) for entry in anytime]
    times = [entry.get("time", 0.0) for entry in anytime]
    return objectives, times


(data, target), mapper = load_adult(scale=True)
X_train, X_test, y_train, y_test = train_test_split(
    data,
    target,
    test_size=0.2,
    random_state=random_state,
)

rf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=random_state,
)
rf.fit(X_train, y_train)

print("RF train acc= ", rf.score(X_train, y_train))
print("RF test acc= ", rf.score(X_test, y_test))

if isinstance(rf, (RandomForestClassifier, AdaBoostClassifier)):
    print(
        "RF size= ",
        sum(tree.tree_.node_count for tree in rf.estimators_),
        " nodes.",
    )

qid = 10
query_frame = X_test.iloc[[qid]]
query_series = query_frame.iloc[0]
query = query_series.to_numpy(dtype=float).flatten()
query_pred = int(rf.predict(query_frame)[0])
print("Query: ", query_series, "(class ", query_pred, ")")

cp_model = ConstraintProgrammingExplainer(rf, mapper=mapper)
start_ = time.time()
cp_explanation = cp_model.explain(
    query,
    y=1 - query_pred,
    norm=1,
    return_callback=True,
    num_workers=num_workers,
    random_seed=random_state,
    max_time=timeout,
    verbose=False,
)
cp_time = time.time() - start_
cp_cf = cp_explanation.to_numpy() if cp_explanation is not None else None

if cp_explanation is not None and cp_cf is not None:
    cp_pred = int(rf.predict(to_frame(cp_cf, data.columns))[0])
    print("CP : ", cp_explanation, "(class ", cp_pred, ")")
else:
    print("CP: No CF found.")

if print_paths:
    print_close_threshold_paths(
        rf,
        cp_cf,
        columns=data.columns,
        query_pred=query_pred,
        label="CP",
    )

milp_model = MixedIntegerProgramExplainer(rf, mapper=mapper)
start_ = time.time()
milp_explanation = milp_model.explain(
    query,
    y=1 - query_pred,
    norm=1,
    return_callback=True,
    num_workers=num_workers,
    random_seed=random_state,
    max_time=timeout,
)
milp_time = time.time() - start_
milp_cf = milp_explanation.to_numpy() if milp_explanation is not None else None

if milp_explanation is not None and milp_cf is not None:
    milp_pred = int(rf.predict(to_frame(milp_cf, data.columns))[0])
    print("MILP : ", milp_explanation, "(class ", milp_pred, ")")
else:
    print("MILP: No CF found.")

if print_paths:
    print_close_threshold_paths(
        rf,
        milp_cf,
        columns=data.columns,
        query_pred=query_pred,
        label="MILP",
    )

print(f"Runtime: CP {cp_time:.3f} s, MILP {milp_time:.3f} s")
print(
    f"Distance: CP {cp_model.get_objective_value():.10f}, "
    f"MILP {milp_model.get_objective_value():.10f}"
)
print(
    f"Status: CP {cp_model.get_solving_status()}, "
    f"MILP {milp_model.get_solving_status()}"
)

if plot_anytime_distances:
    cp_objectives, cp_times = unpack_anytime(cp_model.get_anytime_solutions())
    milp_objectives, milp_times = unpack_anytime(
        milp_model.get_anytime_solutions()
    )

    plt.plot(milp_times, milp_objectives, marker="x", label="MILP", c="b")
    if milp_times and milp_model.get_solving_status() == "OPTIMAL":
        plt.plot(
            milp_times[-1],
            milp_objectives[-1],
            marker="*",
            c="b",
            markersize=15,
        )

    plt.plot(cp_times, cp_objectives, marker="x", label="CP", c="r")
    if cp_times and cp_model.get_solving_status() == "OPTIMAL":
        plt.plot(
            cp_times[-1],
            cp_objectives[-1],
            marker="*",
            c="r",
            markersize=15,
        )

    plt.legend()
    plt.ylabel("CF distance from query")
    plt.xlabel("Running time (second)")
    plt.title("Anytime CF distance comparison.")
    plt.savefig("./anytime_distances_cp_vs_milp.pdf")
