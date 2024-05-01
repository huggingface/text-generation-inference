import json


def main():
    with open("./ShareGPT_V3_unfiltered_cleaned_split.json", "r") as f:
        data = json.load(f)

    # Select only the first 2k conversations that start with a human.
    max = 2000
    conversations = []
    for conversation in data:
        conv = conversation.get("conversations")
        if conv and conv[0]["from"] == "human":
            # Trim the rest of the output
            conversation["conversations"] = conversation["conversations"][:1]
            conversations.append(conversation)

            if len(conversation) >= max:
                break

    with open("./small.json", "w") as f:
        data = json.dump(conversations, f, indent=4)


if __name__ == "__main__":
    main()
