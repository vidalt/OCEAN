import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ocean import ConstraintProgrammingExplainer, MixedIntegerProgramExplainer
from ocean.datasets import load_adult

print_paths = True
plot_anytime_distances = True
num_workers = 8  # Both CP and MILP solving support multithreading
random_state = 0
timeout = 40  # Maximum running time given to the (CP or MILP) solver

# Load the adult dataset
(data, target), mapper = load_adult(
    scale=True
)  # scale=True to perform normalization
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=random_state
)

# Train a RF
rf = RandomForestClassifier(
    n_estimators=10, max_depth=6, random_state=random_state
)

rf.fit(X_train, y_train)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
# Plot the first tree of the forest
plt.figure(figsize=(20,10))
plot_tree(rf.estimators_[0], filled=True)
plt.title("First tree of the Random Forest")
plt.savefig("./first_tree_rf.png")
plt.close()
# print the threshold values for feature 25 of the rf RandomForestClassifier
liste_thresholds = []
for tree in rf.estimators_:
    liste_thresholds.extend(tree.tree_.threshold[tree.tree_.feature == 25])
print("Tree thresholds for feature 25:", sorted(liste_thresholds) )

print("RF train acc= ", rf.score(X_train, y_train))
print("RF test acc= ", rf.score(X_test, y_test))
print(
    "RF size= ",
    sum(a_tree.tree_.node_count for a_tree in rf.estimators_),
    " nodes.",
)

# Define a CF query using the qid-th element of the test set
#qid = 1
#query = X_test.iloc[qid]
import numpy as np 
qid = 10
query = X_test.iloc[qid]
query_pred = rf.predict([np.asarray(query)])[0]
print("Query: ", query, "(class ", query_pred, ")")

# Use the MILP formulation to generate a CF
milp_model = MixedIntegerProgramExplainer(rf, mapper=mapper)
#print("milp_model._num_epsilon", milp_model._num_epsilon)
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
cf = explanation_ocean
#cf[4] += 0.0001
if explanation_ocean is not None:
    print(
        "MILP : ",
        explanation_ocean,
        "(class ",
        rf.predict([explanation_ocean.to_numpy()])[0],
        ")",
    )
    #print("MILP Sollist = ", milp_model.get_anytime_solutions())
else:
    print("MILP: No CF found.")

# debug MILP -------------------------------------------------------
if print_paths:
    cf = explanation_ocean.to_numpy()
    if cf is not None:
        if rf.predict([cf])[0] == query_pred:
            print("INVALID MILP CF : decision path of the CF found by MILP")
            for i, clf in enumerate(rf.estimators_):
                if clf.predict([cf])[0] == query_pred:
                    n_nodes = clf.tree_.node_count
                    children_left = clf.tree_.children_left
                    children_right = clf.tree_.children_right
                    feature = clf.tree_.feature
                    threshold = clf.tree_.threshold
                    values = clf.tree_.value

                    node_indicator = clf.decision_path([cf])
                    leaf_id = clf.apply([cf])

                    sample_id = 0
                    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
                    node_index = node_indicator.indices[
                        node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
                    ]

                    print("[Tree {i}] Rules used to predict sample {id} with features values close to threshold:\n".format(i=i, id=sample_id))
                    for node_id in node_index:
                        # continue to the next node if it is a leaf node
                        if leaf_id[sample_id] == node_id:
                            continue

                        # check if value of the split feature for sample 0 is below threshold
                        if cf[feature[node_id]] <= threshold[node_id]:
                            threshold_sign = "<="
                        else:
                            threshold_sign = ">"
                        if np.abs(cf[feature[node_id]] - threshold[node_id]) < 1e-3:
                            print(
                                "decision node {node} : (cf[{feature}] = {value}) "
                                "{inequality} {threshold})".format(
                                    node=node_id,
                                    sample=sample_id,
                                    feature=feature[node_id],
                                    value=cf[feature[node_id]],
                                    inequality=threshold_sign,
                                    threshold=threshold[node_id],
                                )
                            )
        else:
            print("MILP Valid CF.")
# debug MILP -------------------------------------------------------


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
    #print("CP Sollist = ", cp_model.get_anytime_solutions())
else:
    print("CP: No CF found.")

# debug CP -------------------------------------------------------
if print_paths:
    cf = explanation_oceancp.to_numpy()
    if cf is not None:
        if rf.predict([cf])[0] == query_pred:
            print("INVALID CP CF : decision path of the CF found by CP")
            for i, clf in enumerate(rf.estimators_):
                if clf.predict([cf])[0] == query_pred:
                    n_nodes = clf.tree_.node_count
                    children_left = clf.tree_.children_left
                    children_right = clf.tree_.children_right
                    feature = clf.tree_.feature
                    threshold = clf.tree_.threshold
                    values = clf.tree_.value

                    node_indicator = clf.decision_path([cf])
                    leaf_id = clf.apply([cf])
                    sample_id = 0
                    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
                    node_index = node_indicator.indices[
                        node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
                    ]
                    print(node_index)
                    print("[Tree {i}] Rules used to predict sample {id} with features values close to threshold:\n".format(i=i, id=sample_id))
                    for node_id in node_index:
                        # continue to the next node if it is a leaf node
                        if leaf_id[sample_id] == node_id:
                            continue

                        # check if value of the split feature for sample 0 is below threshold
                        if cf[feature[node_id]] <= threshold[node_id]:
                            threshold_sign = "<="
                        else:
                            threshold_sign = ">"
                        if np.abs(cf[feature[node_id]] - threshold[node_id]) < 1e-3:
                            print(
                                "decision node {node} : (cf[{feature}] = {value}) "
                                "{inequality} {threshold})".format(
                                    node=node_id,
                                    sample=sample_id,
                                    feature=feature[node_id],
                                    value=cf[feature[node_id]],
                                    inequality=threshold_sign,
                                    threshold=threshold[node_id],
                                )
                            )
        else:
            print("CP Valid CF.")
# debug CP -------------------------------------------------------

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
