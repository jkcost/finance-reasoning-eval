import json

with open(
    r"data\financereasoning\raw\FinanceReasoning\hard.json", encoding="utf-8"
) as f:
    data = json.load(f)

print(f"Total items: {len(data)}")
print(f"\nFirst item ground_truth type: {type(data[0].get('ground_truth'))}")
print(f"First item ground_truth value: {data[0].get('ground_truth')}")
