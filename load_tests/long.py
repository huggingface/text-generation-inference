import datasets
import json


dataset = datasets.load_dataset("ccdv/govreport-summarization")
max_new_tokens = 50


conversations = []

for i, item in enumerate(dataset["test"]):
    report = item["report"]

    messages = [{"from": "human", "value": f"Summarize this report: ```{report}```"}]

    conversations.append({"conversations": messages})

with open("long.json", "w") as f:
    json.dump(conversations, f, indent=4)
