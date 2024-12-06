# https://www.gutenberg.org/cache/epub/103/pg103.txt
from openai import OpenAI
import os
import requests

if not os.path.exists("pg103.txt"):
    response = requests.get("https://www.gutenberg.org/cache/epub/103/pg103.txt")
    with open("pg103.txt", "w") as f:
        f.write(response.text)


length = 130000
with open("pg103.txt", "r") as f:
    data = f.read()

messages = [{"role": "user", "content": data[: length * 4]}]

client = OpenAI(base_url="http://localhost:8000/v1", api_key="w")

completion = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct", messages=messages, max_tokens=2
)
