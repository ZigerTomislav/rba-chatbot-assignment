import requests
from models import PromptRequest, PromptResponse
import os
from dotenv import load_dotenv
from util import ChatbotService, DataHandler


url = "http://127.0.0.1:8000/prompt"
url_online = "https://rba-chatbot-assignment.onrender.com"
url_canon = ""

load_dotenv()

API_KEY_VALUE = os.getenv("API_KEY_VALUE")

headers = {
    "X-API-KEY": API_KEY_VALUE
}

test_message = "Koliko košta ulaznica?"
data = {
    "message": test_message
}

response = requests.post(url, headers=headers, json=data)

json_data = response.json()

prompt_response = PromptResponse(**response.json())

#print("intent: ", prompt_response.intent)

service = ChatbotService(use_local=True)
#print(service.get_intent("koje je radno vrijeme?"))

data_handler = DataHandler()
#data_handler.show()

x,y = data_handler.get_test_data_xy()

count = 0
for message, intent in zip(x, y):
    label = service.get_intent(message)

    if label == intent:
        print(message)
        count += 1

print(count)
print("Točnost na test setu", count / len(x) * 100, "%")