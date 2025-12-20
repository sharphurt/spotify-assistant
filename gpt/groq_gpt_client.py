import logging
import os

from groq import Groq
import requests

from gpt.abstract_gpt_client import AbstractGptClient

logger = logging.getLogger(__name__)


class GroqGtpClient(AbstractGptClient):
    def __init__(self, model_name="openai/gpt-oss-120b"):
        super().__init__(model_name)
        logger.info(f"With GROQ model: {model_name}")

        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)

    def request_gpt(self, user_input: str, system_prompt: str):
        logger.info(f"GPT requested: {user_input}")
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
            model=self.model_name,
        )

        logger.info(f"GPT response: {response}")
        if response is not None and len(response.choices) > 0:
            return response.choices[0].message.content

        return None


class GroqProxyClient(AbstractGptClient):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.proxy_url = "http://77.110.122.194:8200/proxy/groq"
        logger.info(f"With proxied GROQ model: {model_name}")

    def request_gpt(self, user_input: str, system_prompt: str):
        data = {
            "user_input": user_input,
            "system_prompt": system_prompt,
            "model_name": self.model_name
        }

        logger.info(f"GPT requested: {user_input}")
        response = requests.post(url=self.proxy_url, json=data)
        if not response.ok:
            logger.error(response)
            return None

        logger.info(f"GPT response: {response.text}")
        return response.text