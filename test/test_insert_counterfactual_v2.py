import json

with open("data/counterfactual/qwen3_logiqa_counterfactual.jsonl", "r", encoding="utf-8") as f:
    data_list = [json.loads(line) for line in f]

with open("test/test.txt", "w", encoding="utf-8") as f:
    output = data_list[0]['counterfactual']
    f.write(output)
