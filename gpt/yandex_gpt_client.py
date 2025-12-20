import os

import openai
from dotenv import load_dotenv

class YandexGptClient:
    def __init__(self, model_name):
        load_dotenv()
        api_key = os.getenv("YANDEX_CLOUD_API_KEY")
        self.model_name = model_name
        self.folder_id = os.getenv("YANDEX_CLOUD_FOLDER")
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://rest-assistant.api.cloud.yandex.net/v1",
            project=self.folder_id
        )

    def request_gpt(self, user_input: str, system_prompt: str):
        response = self.client.responses.create(
            model=f"gpt://{self.folder_id}/{self.model_name}",
            temperature=0.3,
            instructions=system_prompt,
            input=user_input,
            max_output_tokens=1000
        )
        print(response)
        # if response.output_text == '' or response.output_text is None:
        #     print(response)

        return response.output_text
