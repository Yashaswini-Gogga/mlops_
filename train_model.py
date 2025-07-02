# train_model.py

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
X, y = load_iris(return_X_y=True)

# Train the model
clf = LogisticRegression(max_iter=200)
clf.fit(X, y)

# Save the trained model
joblib.dump(clf, 'model.pkl')

print("✅ Model trained and saved as model.pkl")
