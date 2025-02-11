from sklearn.ensemble import RandomForestClassifier

from ocean import MixedIntegerProgramExplainer
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
model = MixedIntegerProgramExplainer(rf, mapper=mapper)
x = x.to_numpy().flatten()
explanation = model.explain(x, y=1 - y, norm=1)

# Show the explanation
print(explanation)
