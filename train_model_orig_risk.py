import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
df = pd.read_csv("data/engineered_incident_data.csv")

# Define features and target for risk classification
features = ["priority", "impact", "urgency", "reassignment_count", "reopen_count", "time_to_resolution", "complexity_score"]
X_class = df[features]
y_class = df["sla_breach"]  # Predict if SLA will be breached (1 = High Risk)

# Split data (80% train, 20% test)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

# Handle Imbalanced Data for Risk Classification
ros = RandomOverSampler(random_state=42)
X_train_class_resampled, y_train_class_resampled = ros.fit_resample(X_train_class, y_train_class)

# Train Risk Prediction Model
clf = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=3,  # Reduce complexity
    min_samples_split=20,  # Increase required samples per split
    min_samples_leaf=10,  # Larger leaf size to generalize better
    random_state=42
)
clf.fit(X_train_class_resampled, y_train_class_resampled)

# Evaluate Classification Model
y_pred_class = clf.predict(X_test_class)
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"Final Risk Prediction Model Accuracy: {accuracy:.2f}")

# Save the Risk Prediction Model
with open("models/risk_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Risk Model saved successfully!")
