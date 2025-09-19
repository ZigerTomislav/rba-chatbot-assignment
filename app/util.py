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
        self.test_data = pd.read_csv("dataset_test_hard.csv", skipinitialspace=True)
        self.train_data = pd.read_csv("dataset_train.csv", skipinitialspace=True)
        self.test_data_easy = pd.read_csv("dataset_test_easy.csv", skipinitialspace=True)

    def show(self):
        print(self.test_data.head())
        print(self.train_data.head())

    def get_test_data_hard_xy(self):
        return self.get_data_xy(self.test_data)

    def get_test_data_easy_xy(self):
        return self.get_data_xy(self.test_data_easy)

    def get_train_data_xy(self):
        return self.get_data_xy(self.train_data)

    def get_combined_data_xy(self):
        combined_data = pd.concat([self.train_data, self.test_data, self.test_data_easy], ignore_index=True)
        return self.get_data_xy(combined_data)

    def get_data_xy(self, data):
        x = data["message"]
        y = data["intent"]
        return x, y


class DataAugmenter:

    def remove_first(self, X, Y):

        new_X, new_Y = [], []
        for i, j in zip(X, Y):
            if len(i) > 1:
                new_X.append(i[1:])
                new_Y.append(j)

        return pd.Series(new_X), pd.Series(new_Y)

    def remove_one(self, X, Y):
        new_X, new_Y = [], []
        for i, j in zip(X, Y):
            for k in range(len(i)):
                before = i[:k]
                after = i[k+1:]

                new_string = before + after
                new_X.append(new_string)
                new_Y.append(j)

        return pd.Series(new_X), pd.Series(new_Y)

    def remove_croatian(self, X, Y):
        replace_dict = {"č": "c", "ć": "c", "đ": "d", "ž": "z", "š": "s"}
        new_X, new_Y = [], []
        for i, j in zip(X, Y):
            new_string = i
            #for v in new_string:
            #    if v in replace_dict.keys():
            #        print("ima hr znakova")
            for k in replace_dict.keys():
                new_string = new_string.replace(k, replace_dict[k])

            if new_string != i:
                new_X.append(new_string)
                new_Y.append(j)

        return pd.Series(new_X), pd.Series(new_Y)



