import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ocean.datasets import load_adult
from ocean.explainer import MIPExplainer

# Load the adult dataset
mapper, (data, target) = load_adult()

# Generate a random instance
generator = np.random.default_rng(42)
i = generator.integers(0, data.shape[0])
x = data.iloc[[i]]

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
rf.fit(data, target)

# Predict the class of the random instance
y = int(rf.predict(x)[0])

# Explain the prediction using MIPEXplainer
model = MIPExplainer(rf, mapper=mapper)
model.set_majority_class(m_class=1 - y)
model.optimize()

# Show the explanation
print(model.solution)
