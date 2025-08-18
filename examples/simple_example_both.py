from ocean import MixedIntegerProgramExplainer,ConstraintProgrammingExplainer
from ocean.datasets import load_adult
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time 
gurobi_statuses = {1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT", 8: "NODE_LIMIT", 9: "TIME_LIMIT", 10: "SOLUTION_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC", 13: "SUBOPTIMAL", 14: "INPROGRESS", 15: "USER_OBJ_LIMIT", 16: "WORK_LIMIT"}  # see https://www.gurobi.com/documentation/9.5/refman/optimization_status_codes.html#sec:StatusCodes


# Load the adult dataset
(data, target), mapper = load_adult(scale=True) # scale=True to perform normalization
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

# Train a RF
rf = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=42)
rf.fit(X_train, y_train)
print("RF train acc= ", rf.score(X_train, y_train))
print("RF test acc= ", rf.score(X_test, y_test))
print("RF size= ", sum([a_tree.tree_.node_count for a_tree in rf.estimators_]), " nodes.")

# Define a CF query using the qid-th element of the test set
qid = 0
query = X_test.iloc[qid]
query_pred = rf.predict([query.to_numpy()])[0]
print("Query: ", query, "(class ", query_pred, ")")

# Use the MILP formulation to generate a CF
model = MixedIntegerProgramExplainer(rf, mapper=mapper)

start_=time.time()
explanation_ocean = model.explain(query, y=1-query_pred, norm=1, num_workers=4, return_callback=True)
milp_time = time.time() - start_

print("MILP : ", explanation_ocean, "(class ", rf.predict([explanation_ocean.to_numpy()])[0], ")")
print("MILP Sollist = ", model.callback.sollist)

# Use the CP formulation to generate a CF
cpmodel = ConstraintProgrammingExplainer(rf, mapper=mapper)

start_=time.time()
explanation_oceancp = cpmodel.explain(query, y=1-query_pred, norm=1, num_workers=4, return_callback=True)
cp_time = time.time() - start_

print("CP : ", explanation_oceancp, "(class ", rf.predict([explanation_oceancp.to_numpy()])[0], ")")
print("CP Sollist = ", cpmodel.callback.sollist) # To be divided by the scaling factor (normally 1e8)

# Display summary statistics
print("Runtime: CP %.3f s, MILP %.3f s" %(cp_time, milp_time))
print("Distance: CP %.10f, MILP %.10f" %(cpmodel.solver.objective_value/1e8, model.ObjVal))
print("Status: CP %s, MILP %s" %(cpmodel.Status, gurobi_statuses[model.Status]))