import requests
from models import PromptRequest, PromptResponse
import os
from dotenv import load_dotenv
import pandas as pd

class ChatbotService:

    url_local = "http://127.0.0.1:8000/prompt"
    url_online = "https://rba-chatbot-assignment.onrender.com"
    api_url = url_online
    api_key_value = None

    def __init__(self, use_local=True):

        if use_local:
            self.api_url = self.url_local

        load_dotenv()
        self.api_key_value = os.getenv("API_KEY_VALUE")
        if self.api_key_value is None:
            print("API_KEY_VALUE not set")

    def get_intent(self, message):
        response = self.make_request(message)
        prompt_response = PromptResponse(**response.json())
        return prompt_response.intent

    def get_full_response(self, message):
        response = self.make_request(message)
        prompt_response = PromptResponse(**response.json())
        return prompt_response

    def make_request(self, message):

        data = {
            "message": message
        }
        headers = {
            "X-API-KEY": self.api_key_value
        }

        response = requests.post(self.api_url, headers=headers, json=data)
        return response

