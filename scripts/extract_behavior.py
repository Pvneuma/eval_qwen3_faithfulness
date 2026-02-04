import json

INPUT_FILE = "data/counterfactual/behavior.jsonl"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

output = []
for item in data:
    id = item['id']
    # 提取result字符串中</think>和<|im_end|>之间的内容
    raw_text = item['result']
    start_tag = "</think>"
    end_tag = "<|im_end|>"

    start_idx = raw_text.find(start_tag)
    end_idx = raw_text.find(end_tag)

    if start_idx != -1:
        # 如果找到了 </think>，从它后面开始截取
        content = raw_text[start_idx + len(start_tag):]
        # 如果之后还有 <|im_end|>，则截取到它之前
        if end_tag in content:
            content = content[:content.find(end_tag)]
        result = content.strip()
    # 判断result中是否包含CONSISTENTLY_FOLLOWED或EXPLICITLY_CORRECTED中的一个，如果有就将其赋值给result，如果没有就报错并打印当前id，并且确保这两个值没有同时出现
    if "EXPLICITLY_CORRECTED" in result and "CONSISTENTLY_FOLLOWED" in result:
        print(f"Error: Multiple judgments found in result for id {id}")
    if "EXPLICITLY_CORRECTED" in result:
        result = "EXPLICITLY_CORRECTED"
    elif "CONSISTENTLY_FOLLOWED" in result:
        result = "CONSISTENTLY_FOLLOWED"
    else:
        print(f"Error: No valid judgment found in result for id {id}")
        exit(1)

    data = {"id": id, "result": result}
    output.append(data)

with open("data/counterfactual/behavior_result.jsonl", "w", encoding="utf-8") as f:
    for item in output:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
