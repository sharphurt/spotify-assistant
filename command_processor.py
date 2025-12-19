import re

from llama_cpp import Llama

SYSTEM_PROMPT = open('system_prompt.txt', 'r', encoding='utf-8').read()
FORBIDDEN_STRINGS = re.compile(r'(\{|}|\[|]|```|<script>|</|role:|system:|assistant:)')


class CommandProcessor:
    def __init__(self):
        self.llm = Llama(
            model_path="./models/Llama-3.1-8B-Instruct-Q3_K_L.gguf",
            n_gpu_layers=-1,
            n_ctx=2048,
            n_threads=8
        )

    def decode_command(self, input: str):
        output = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.sanitaze_input(input)},
            ]
        )

        print(output)

    def sanitaze_input(self, input):
        return re.sub(FORBIDDEN_STRINGS, '', input)
