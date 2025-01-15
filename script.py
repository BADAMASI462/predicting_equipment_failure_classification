import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic classification data
data_class = {
    'Usage_Hours': np.random.uniform(100, 1000, 100),
    'Maintenance_Score': np.random.uniform(0, 1, 100),
    'Operational_Temperature': np.random.uniform(30, 150, 100),
    'Failure': np.random.choice([0, 1], 100)  # 0: No Failure, 1: Failure
}

df_class = pd.DataFrame(data_class)
X_class = df_class[['Usage_Hours', 'Maintenance_Score', 'Operational_Temperature']]
y_class = df_class['Failure']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_c, y_train_c)

# Predict and evaluate
y_pred_c = rf_model.predict(X_test_c)
print("\nClassification Results:")
print(f"Accuracy: {accuracy_score(y_test_c, y_pred_c)}")
print(classification_report(y_test_c, y_pred_c))
