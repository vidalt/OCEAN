import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ocean import ConstraintProgrammingExplainer, MixedIntegerProgramExplainer
from ocean.datasets import load_adult

plot_anytime_distances = True
num_workers = 8  # Both CP and MILP solving support multithreading
random_state = 42
timeout = 60  # Maximum running time given to the (CP or MILP) solver

# Load the adult dataset
(data, target), mapper = load_adult(
    scale=True
)  # scale=True to perform normalization
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, random_state=random_state
)

# Train a RF
rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, random_state=random_state
)
rf.fit(X_train, y_train)
print("RF train acc= ", rf.score(X_train, y_train))
print("RF test acc= ", rf.score(X_test, y_test))
print(
    "RF size= ",
    sum(a_tree.tree_.node_count for a_tree in rf.estimators_),
    " nodes.",
)

# Define a CF query using the qid-th element of the test set
qid = 1
query = X_test.iloc[qid]
query_pred = rf.predict([query.to_numpy()])[0]
print("Query: ", query, "(class ", query_pred, ")")

# Use the MILP formulation to generate a CF
milp_model = MixedIntegerProgramExplainer(rf, mapper=mapper)

start_ = time.time()
explanation_ocean = milp_model.explain(
    query,
    y=1 - query_pred,
    norm=1,
    return_callback=True,
    num_workers=num_workers,
    random_seed=random_state,
    max_time=timeout,
)
milp_time = time.time() - start_
if explanation_ocean is not None:
    print(
        "MILP : ",
        explanation_ocean,
        "(class ",
        rf.predict([explanation_ocean.to_numpy()])[0],
        ")",
    )
    print("MILP Sollist = ", milp_model.get_anytime_solutions())
else:
    print("MILP: No CF found.")

# Use the CP formulation to generate a CF
cp_model = ConstraintProgrammingExplainer(rf, mapper=mapper)

start_ = time.time()
explanation_oceancp = cp_model.explain(
    query,
    y=1 - query_pred,
    norm=1,
    return_callback=True,
    num_workers=num_workers,
    random_seed=random_state,
    max_time=timeout,
)
cp_time = time.time() - start_

if explanation_oceancp is not None:
    print(
        "CP : ",
        explanation_oceancp,
        "(class ",
        rf.predict([explanation_oceancp.to_numpy()])[0],
        ")",
    )
    print("CP Sollist = ", cp_model.get_anytime_solutions())
else:
    print("CP: No CF found.")

# Display summary statistics
print(f"Runtime: CP {cp_time:.3f} s, MILP {milp_time:.3f} s")
print(
    f"Distance: CP {cp_model.get_objective_value():.10f},",
    f" MILP {milp_model.get_objective_value():.10f}",
)
print(
    f"Status: CP {cp_model.get_solving_status()},",
    f" MILP {milp_model.get_solving_status()}",
)


if plot_anytime_distances:
    import matplotlib.pyplot as plt

    anytime_solution = {}
    anytime_solution["cp"] = cp_model.get_anytime_solutions()
    anytime_solution["mip"] = milp_model.get_anytime_solutions()
    cpobjectives = []
    cptimes = []
    for dic in anytime_solution.get("cp", []):
        cpobjectives.append(dic.get("objective_value", 0))
        cptimes.append(dic.get("time", 0))
    milpobjectives = []
    milptimes = []
    for dic in anytime_solution.get("mip", []):
        milpobjectives.append(dic.get("objective_value", 0))
        milptimes.append(dic.get("time", 0))

    plt.plot(milptimes, milpobjectives, marker="x", label="MILP", c="b")
    if milp_model.get_solving_status() == "OPTIMAL":
        plt.plot(
            milptimes[-1], milpobjectives[-1], marker="*", c="b", markersize=15
        )

    plt.plot(cptimes, cpobjectives, marker="x", label="CP", c="r")
    if cp_model.get_solving_status() == "OPTIMAL":
        plt.plot(
            cptimes[-1], cpobjectives[-1], marker="*", c="r", markersize=15
        )

    plt.legend()
    plt.ylabel("CF distance from query")
    plt.xlabel("Running time (second)")

    plt.title("Anytime CF distance comparison.")
    plt.savefig("./anytime_distances_cp_vs_milp.pdf")
