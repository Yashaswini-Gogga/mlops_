import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'income']
df = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)
df.dropna(inplace=True)

# Encode target and sensitive attribute
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
df = pd.get_dummies(df, drop_first=True)

X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Original Model:")
print(classification_report(y_test, y_pred))

# Check accuracy by sex
X_test['sex'] = X_test['sex']
X_test_copy = X_test.copy()
X_test_copy['actual'] = y_test
X_test_copy['prediction'] = y_pred
grouped = X_test_copy.groupby('sex').apply(lambda g: accuracy_score(g['actual'], g['prediction']))
print("Accuracy by sex (original):\n", grouped)

# Mitigation: reweighting
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
model_rw = LogisticRegression(max_iter=1000)
model_rw.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_rw = model_rw.predict(X_test)

print("\nReweighted Model:")
print(classification_report(y_test, y_pred_rw))

# Accuracy by sex after reweighting
X_test_copy['prediction_rw'] = y_pred_rw
grouped_rw = X_test_copy.groupby('sex').apply(lambda g: accuracy_score(g['actual'], g['prediction_rw']))
print("Accuracy by sex (reweighted):\n", grouped_rw)
