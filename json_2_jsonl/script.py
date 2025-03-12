import json

# Load the original JSON file
with open("input.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract messages
messages = data["messages"]

# Convert to JSONL format
with open("output.jsonl", "w", encoding="utf-8") as f:
    for message in messages:
        jsonl_entry = {
            "messages": [message]
        }
        f.write(json.dumps(jsonl_entry) + "\n")

print("Conversion completed! JSONL file saved as output.jsonl")
