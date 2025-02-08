import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ocean.datasets import load_adult
from ocean.explainer import MIPExplainer

# Load the adult dataset
mapper, (data, target) = load_adult()

# Generate a random instance from the dataset.
generator = np.random.default_rng(42)
x = generator.choice(data, size=1)

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
rf.fit(data, target)

# Predict the class of the random instance
y = int(rf.predict(x).item())

# Explain the prediction using MIPEXplainer
model = MIPExplainer(rf, mapper=mapper)
x = x.flatten()
explanation = model.explain(x, y=1 - y, norm=1)

# Show the explanation
print(explanation)
