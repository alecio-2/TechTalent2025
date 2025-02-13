import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# Load the dataset
df = pd.read_csv("data/engineered_incident_data.csv")

# Define features for time to resolution prediction
features = ["priority", "impact", "urgency", "reassignment_count", "reopen_count", "complexity_score"]
X_reg = df[features]

# 🚨 Square Root Transformation for Stability
df["time_to_resolution"] = np.sqrt(df["time_to_resolution"] + 50)
y_reg = df["time_to_resolution"]

# Split data (80% train, 20% test)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# ✅ Use `MinMaxScaler` for Stability
scaler_features = MinMaxScaler()
X_train_reg_scaled = scaler_features.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_features.transform(X_test_reg)

scaler_target = MinMaxScaler()
y_train_reg_scaled = scaler_target.fit_transform(y_train_reg.values.reshape(-1, 1)).ravel()
y_test_reg_scaled = scaler_target.transform(y_test_reg.values.reshape(-1, 1)).ravel()

# Save scalers
with open("models/scaler_features.pkl", "wb") as f:
    pickle.dump(scaler_features, f)

with open("models/scaler_target.pkl", "wb") as f:
    pickle.dump(scaler_target, f)

# Train Resolution Time Prediction Model
reg = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    gamma=0.1,
    colsample_bytree=0.9,
    subsample=0.8,
    objective="reg:squarederror",
    random_state=42
)
reg.fit(X_train_reg_scaled, y_train_reg_scaled)

# Evaluate Regression Model
y_pred_reg_scaled = reg.predict(X_test_reg_scaled)
y_pred_reg = np.square(scaler_target.inverse_transform(y_pred_reg_scaled.reshape(-1, 1))).ravel()
y_pred_reg = [max(0, val) for val in y_pred_reg]

mae = mean_absolute_error(y_test_reg, y_pred_reg)
print(f"Final Time to Resolution MAE: {mae:.2f} hours")

# Save the Resolution Time Prediction Model
with open("models/resolution_model.pkl", "wb") as f:
    pickle.dump(reg, f)

print("✅ Resolution Model saved successfully!")

# ✅ Histogram to check transformation
plt.hist(df["time_to_resolution"], bins=50, edgecolor="black")
plt.xlabel("Square Root Transformed Resolution Time (Adjusted)")
plt.ylabel("Count")
plt.title("Distribution of Incident Resolution Times (Square Root Transformed, Adjusted)")
plt.show()
