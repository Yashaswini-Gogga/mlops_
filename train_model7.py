from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
X, y = load_iris(return_X_y=True)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model to file
joblib.dump(model, 'model.pkl')

print("Model trained and saved successfully as model.pkl")
