import streamlit as st
import requests
import json
import random
import pandas as pd

# API Endpoints
PREDICT_URL = "http://127.0.0.1:8000/predict/"
ASK_URL = "http://127.0.0.1:8000/ask/"

# Load Incidents
with open("data/incidents.json", "r") as f:
    incidents = json.load(f)

# Convert incidents to DataFrame for table display
df_incidents = pd.DataFrame(incidents)

# Streamlit UI
st.title("      🚨   ARIS   🚨")
st.title("Automated Risk Insight System")
st.write("This app analyzes IT incidents, predicts risks, resolution times, and provides AI recommendations.")

# Display all incidents in a table
st.subheader("📋 List of Incidents")
st.dataframe(df_incidents)

# Select a random incident
incident = random.choice(incidents)

# Show the randomly selected incident
st.subheader("🎯 Selected Incident for Analysis")
st.json(incident)

# Remove time_to_resolution from the input data before sending it to the API
incident_payload = incident.copy()
incident_payload.pop("time_to_resolution", None)

# Send the incident to the API for prediction
st.subheader("📊 AI Analysis")
response = requests.post(PREDICT_URL, json=incident_payload)

if response.status_code == 200:
    result = response.json()
    risk_category = result["risk_category"]
    predicted_time = result["predicted_resolution_time"]

    st.success(f"**Risk Category:** {risk_category}")
    st.info(f"**Predicted Resolution Time:** {predicted_time} hours")
else:
    st.error("Error analyzing incident. Please check API status.")

# Chatbox for LLM Recommendations
st.subheader("💬 AI Chat Assistant")

# Initial LLM Response
ask_payload = {
    "question": "What should the user do about this incident?",
    "incident": incident
}
ask_response = requests.post(ASK_URL, json=ask_payload)

if ask_response.status_code == 200:
    llm_response = ask_response.json()["response"]
    st.write("🤖 **AI Recommendation:**")
    st.success(llm_response)
else:
    st.error("Error getting LLM response.")

# User input for chat
user_question = st.text_input("Type your question about this incident:")
if st.button("Ask AI"):
    if user_question:
        ask_payload["question"] = user_question
        ask_response = requests.post(ASK_URL, json=ask_payload)

        if ask_response.status_code == 200:
            st.write("🤖 **AI Response:**")
            st.success(ask_response.json()["response"])
        else:
            st.error("Error getting LLM response.")
