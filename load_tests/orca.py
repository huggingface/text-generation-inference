import json
import datasets
import tqdm


def main():
    dataset = datasets.load_dataset("Open-Orca/OpenOrca", split="train")
    # Select only the first 2k conversations that start with a human.
    max = min(2000, len(dataset))
    conversations = []
    for item in tqdm.tqdm(dataset, total=max):
        conversation = {
            "conversations": [
                {"from": "human", "value": item["question"]},
            ],
            "id": item["id"],
        }
        conversations.append(conversation)
        if len(conversations) >= max:
            break

    with open("./small.json", "w") as f:
        data = json.dump(conversations, f, indent=4)


if __name__ == "__main__":
    main()
