from sklearn.ensemble import RandomForestClassifier

from ocean import (
    ConstraintProgrammingExplainer,
    MaxSATExplainer,
    MixedIntegerProgramExplainer,
)
from ocean.datasets import load_adult

# Load the adult dataset
(data, target), mapper = load_adult(scale=True)

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
rf.fit(data, target)

# Select an instance to explain from the dataset
x = data.iloc[0].to_frame().T
x_np = x.to_numpy().flatten()

# Predict the class of the instance
y_pred = int(rf.predict(x).item())
target_class = 1 - y_pred  # Binary classification - choose opposite class

print(f"Instance shape: {x_np.shape}")
print(f"Original prediction: {y_pred}")
print(f"Target counterfactual class: {target_class}")

# Explain the prediction using MaxSATExplainer
print("\n--- MaxSAT Explainer ---")
try:
    maxsat_model = MaxSATExplainer(rf, mapper=mapper)
    maxsat_explanation = maxsat_model.explain(x_np, y=target_class, norm=1)
    if maxsat_explanation is not None:
        cf_np = maxsat_explanation.to_numpy()
        print("MaxSAT CF:", cf_np)
        print("MaxSAT CF prediction:", rf.predict([cf_np])[0])
        print("Objective value:", maxsat_model.get_objective_value())
        print("Status:", maxsat_model.get_solving_status())
    else:
        print("MaxSAT: No counterfactual found.")
except (ValueError, RuntimeError, ImportError) as e:
    import traceback

    print(f"MaxSAT Error: {e}")
    traceback.print_exc()

# Explain the prediction using MIPExplainer
print("\n--- MIP Explainer ---")
try:
    mip_model = MixedIntegerProgramExplainer(rf, mapper=mapper)
    mip_explanation = mip_model.explain(x_np, y=target_class, norm=1)
    if mip_explanation is not None:
        cf_np = mip_explanation.to_numpy()
        print("MIP CF:", cf_np)
        print("MIP CF prediction:", rf.predict([cf_np])[0])
        print("Objective value:", mip_model.get_objective_value())
        print("Status:", mip_model.get_solving_status())
    else:
        print("MIP: No counterfactual found.")
except (ValueError, RuntimeError, ImportError) as e:
    print(f"MIP Error: {e}")

# Explain the prediction using CPExplainer
print("\n--- CP Explainer ---")
try:
    cp_model = ConstraintProgrammingExplainer(rf, mapper=mapper)
    cp_explanation = cp_model.explain(x_np, y=target_class, norm=1)
    if cp_explanation is not None:
        cf_np = cp_explanation.to_numpy()
        print("CP CF:", cf_np)
        print("CP CF prediction:", rf.predict([cf_np])[0])
        print("Objective value:", cp_model.get_objective_value())
        print("Status:", cp_model.get_solving_status())
    else:
        print("CP: No counterfactual found.")
except (ValueError, RuntimeError, ImportError) as e:
    print(f"CP Error: {e}")
