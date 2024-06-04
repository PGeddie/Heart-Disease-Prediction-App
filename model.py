import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

data = {
    'age': [63, 37, 41, 56, 57, 57, 56, 44, 52, 57],
    'sex': [1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    'cp': [3, 2, 1, 1, 0, 0, 1, 1, 2, 2],
    'trestbps': [145, 130, 130, 120, 120, 140, 140, 120, 172, 150],
    'chol': [233, 250, 204, 236, 354, 192, 294, 263, 199, 168],
    'fbs': [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'restecg': [0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    'thalach': [150, 187, 172, 178, 163, 148, 153, 173, 162, 174],
    'exang': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6, 0.4, 1.3, 0.0, 1.5, 1.6],
    'slope': [0, 0, 2, 2, 2, 1, 1, 2, 2, 2],
    'ca': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'thal': [1, 2, 2, 2, 2, 1, 2, 3, 3, 2],
    'target': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

model_filename = 'heart_disease_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
