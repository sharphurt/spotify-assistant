from abc import abstractmethod, ABC


class AbstractGptClient(ABC):

    def __init__(self, model_name="openai/gpt-oss-120b"):
        self.model_name = model_name

    @abstractmethod
    def request_gpt(self, user_input: str, system_prompt: str):
        pass
