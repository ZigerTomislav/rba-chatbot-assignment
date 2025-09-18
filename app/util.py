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

class DataHandler:
    def __init__(self):
        self.test_data = pd.read_csv("dataset_test.csv")
        self.train_data = pd.read_csv("dataset_train.csv")
        self.test_data_easy = pd.read_csv("dataset_test_easy.csv")

    def show(self):
        print(self.test_data.head())
        print(self.train_data.head())

    def get_test_data_xy(self):
        return self.get_data_xy(self.test_data)

    def get_test_data_easy_xy(self):
        return self.get_data_xy(self.test_data_easy)

    def get_train_data_xy(self):
        return self.get_data_xy(self.train_data)

    def get_data_xy(self, data):
        x = data["message"]
        y = data["intent"]
        return x, y


class DataAugmenter:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def remove_first(self, X = None, Y = None):
        if X == None or Y == None:
            print("X or Y empty")
            return

        originalX = X
        #for i in X:




