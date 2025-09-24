from sklearn.ensemble import RandomForestClassifier

from ocean import ConstraintProgrammingExplainer, MixedIntegerProgramExplainer
from ocean.datasets import load_adult

# Load the adult dataset
(data, target), mapper = load_adult()

# Select an instance to explain from the dataset
x = data.iloc[0].to_frame().T

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
rf.fit(data, target)

# Predict the class of the random instance
y = int(rf.predict(x).item())

# Explain the prediction using MIPEXplainer
mip_model = MixedIntegerProgramExplainer(rf, mapper=mapper)
x = x.to_numpy().flatten()
mip_explanation = mip_model.explain(x, y=1 - y, norm=1)

# Explain the prediction using CPEExplainer
cp_model = ConstraintProgrammingExplainer(rf, mapper=mapper)
cp_explanation = cp_model.explain(x, y=1 - y, norm=1)

# Show the explanation
print("MIP", mip_explanation)
print("CP", cp_explanation)
