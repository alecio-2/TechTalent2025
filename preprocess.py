import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from datetime import datetime

# Step 1: Data Preprocessing
# Load the dataset
df = pd.read_csv("data/incident_event_log.csv")

# Display basic information
print("Dataset Shape:", df.shape) # this will print the shape of the dataset
print("Columns:", df.columns) # this will print the columns of the dataset
print("Missing Values:\n", df.isnull().sum()) # this will print the count of missing values in each column

# Step 2: Data Cleaning
# Drop unnecessary columns
drop_cols = ["number", "sys_created_by", "sys_updated_by", "caller_id", "opened_by", "resolved_by", "closed_at"]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

# Step 3: Convert Categorical Features to Numerical 

# Convert categorical features
cat_features = ["incident_state", "impact", "urgency", "priority", "category", "subcategory", "contact_type"]
encoders = {}

for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save the encoders for later use
import pickle
with open("models/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Save the cleaned dataset
df.to_csv("data/cleaned_incident_data.csv", index=False)

print("Data Preprocessing Complete - Saved as cleaned_incident_data.csv")


# Step 4: Feature Engineering
# In this step I will create a new dataset with additional features that can be used to train the model
# The new features will include:
# - Time to resolution (in hours)
# - Incident complexity score (based on reassignment count, reopen count, and priority)
# - Escalation flag based on reassignment count
# - SLA breach flag based on time to resolution

# Load the cleaned dataset
df = pd.read_csv("data/cleaned_incident_data.csv")

# Convert timestamps to datetime format
df["opened_at"] = pd.to_datetime(df["opened_at"], errors='coerce', dayfirst=True)
df["resolved_at"] = pd.to_datetime(df["resolved_at"], errors='coerce', dayfirst=True)

# Calculate time to resolution (in hours)
df["time_to_resolution"] = (df["resolved_at"] - df["opened_at"]).dt.total_seconds() / 3600

# Fill missing resolution times with median
#df["time_to_resolution"].fillna(df["time_to_resolution"].median(), inplace=True)
df["time_to_resolution"] = df["time_to_resolution"].fillna(df["time_to_resolution"].median())

# Create an incident complexity score
df["complexity_score"] = df["reassignment_count"] + df["reopen_count"] + df["priority"]

# Define escalation based on reassignment count
df["escalated"] = df["reassignment_count"].apply(lambda x: 1 if x > 2 else 0)

# Define SLA breach (1 if exceeded, 0 if within limit)
df["sla_breach"] = df["time_to_resolution"].apply(lambda x: 1 if x > 24 else 0)  # Assuming SLA is 24 hours

# Save the new dataset
df.to_csv("data/engineered_incident_data.csv", index=False)

print("Feature Engineering Complete - Saved as engineered_incident_data.csv")