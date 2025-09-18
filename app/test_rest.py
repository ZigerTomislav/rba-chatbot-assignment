import requests
from models import PromptRequest, PromptResponse


url = "http://127.0.0.1:8000/prompt"
url_online = "https://rba-chatbot-assignment.onrender.com"
url_canon = ""

headers = {
    "accept": "application/json",
    "X-API-KEY": "B0T_1N_TH3_B4NK",
    "Content-Type": "application/json",
}

test_message = "Koliko košta ulaznica?"
data = {
    "message": test_message
}

response = requests.post(url, headers=headers, json=data)

json_data = response.json()

prompt_response = PromptResponse(**response.json())


print("intent: ", prompt_response.intent)
