import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

data = {
     'age': [63, 37, 41, 56, 57, 67, 67, 62, 63, 53],
    'sex': [1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
    'cp': [3, 2, 1, 1, 0,  0, 0, 0, 0, 0],
    'trestbps': [145, 130, 130, 120, 120, 160, 120, 140, 130, 140],
    'chol': [233, 250, 204, 236, 354, 286, 229, 268, 254, 203],
    'fbs': [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    'restecg': [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    'thalach': [150, 187, 172, 178, 163, 108, 129, 160, 147, 155],
    'exang': [0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
    'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6, 1.5, 2.6, 3.6, 1.4, 3.1],
    'slope': [0, 0, 2, 2, 2, 1, 1, 0, 1, 1],
    'ca': [0, 0, 0, 0, 0, 3, 2, 2, 1, 0],
    'thal': [1, 2, 2, 2, 2, 2, 3, 2, 3, 3],
    'target': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
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
