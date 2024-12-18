import os
import json


for root, dirs, files in os.walk("."):
    for filename in files:
        if filename.endswith(".json"):
            with open(os.path.join(root, filename), "r") as f:
                data = json.load(f)

            print(os.path.join(root, filename))
            try:
                if filename.endswith("_load.json"):
                    for i in range(len(data)):
                        data[i]["details"]["prefill"] = []
                else:
                    data["details"]["prefill"] = []
            except Exception:
                pass

            with open(os.path.join(root, filename), "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
