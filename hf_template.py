import json
import requests
import os


class Model:
    def __init__(self, model="gpt2"):
        self.url = f"https://api-inference.huggingface.co/models/{model}"
        self.key = os.getenv("hf_api")

    def query(self, query):
        headers = {"Authorization": f"Bearer {self.key}"}
        data = json.dumps(query)
        response = requests.request("POST", self.url, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))[0]["generated_text"]


if __name__ == "__main__":
    m = Model("gpt2")
    print(m.query("I like apples because"))
