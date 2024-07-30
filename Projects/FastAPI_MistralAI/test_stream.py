import json
import os
import requests
import ast
import json
from dotenv import load_dotenv

load_dotenv()

BEAR_TOKEN_API_KEY = os.getenv("BEAR_TOKEN_API_KEY")

DEBUG = True

if DEBUG:
    url = "http://localhost:8000/v1/chat/completions"
else:
    url = "https://api.mistral.ai/v1/chat/completions"


data = {
    "model": "mistral-small-latest",
    "messages": [{"role": "user", "content": "Heloo?"}],
    "stream": True,
}

headers = {"Content-type": "application/json", "Authorization": f"Bearer {BEAR_TOKEN_API_KEY}"}

# print chunk
with requests.post(url, data=json.dumps(data), headers=headers, stream=True) as r:
    for chunk in r.iter_content(1024):
        print(chunk)
        print("\n\n")

# write the streaming to terminal
with requests.post(url, data=json.dumps(data), headers=headers, stream=True) as r:
    for chunk in r.iter_content(1024):
        # print(chunk.decode().split("data: ")[1])
        response_dict = json.loads(chunk.decode().split("data: ")[1])
        content = response_dict["choices"][0]["delta"]["content"]

        print(content, end="")


with requests.post(url, data=json.dumps(data), headers=headers, stream=True) as r:
    for chunk in r.iter_content(1024):
        print(chunk, "")
