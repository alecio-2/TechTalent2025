import json
import requests
import random

# API endpoints
PREDICT_URL = "http://127.0.0.1:8000/predict/"
ASK_URL = "http://127.0.0.1:8000/ask/"

# Load incidents from file
with open("data/incidents.json", "r") as f:
    incidents = json.load(f)

# Select a random incident
incident = random.choice(incidents)

print("\n🔍 Sending incident for analysis:")
print(json.dumps(incident, indent=2))

# Send the incident for analysis (predict resolution time first, then risk)
predict_response = requests.post(PREDICT_URL, json=incident)

if predict_response.status_code == 200:
    result = predict_response.json()
    risk_category = result["risk_category"]
    predicted_time = result["predicted_resolution_time"]

    print("\n📊 Analysis Results:")
    print(f"Risk Category: {risk_category}")
    print(f"Predicted Resolution Time: {predicted_time} hours")

    # Append the predicted resolution time to the incident data
    incident["time_to_resolution"] = predicted_time
else:
    print("\n⚠️ Error analyzing incident. Exiting.")
    exit()


# Automatically send this to the LLM for a response
print("\n🤖 Chatbot Response:")
ask_payload = {"question": "What should the user do about this incident?", "incident": incident}
ask_response = requests.post(ASK_URL, json=ask_payload)

if ask_response.status_code == 200:
    print(ask_response.json()["response"])
else:
    print("⚠️ Error getting LLM response.")

# Interactive chat loop
while True:
    user_input = input("\nType your question about this incident (or 'exit' to quit): ").strip()
    if user_input.lower() == "exit":
        print("👋 Exiting chat.")
        break

    ask_payload["question"] = user_input
    ask_response = requests.post(ASK_URL, json=ask_payload)

    if ask_response.status_code == 200:
        print("\n🤖", ask_response.json()["response"])
    else:
        print("⚠️ Error getting LLM response.")
