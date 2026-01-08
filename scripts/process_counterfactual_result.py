import json
import textwrap

HAS_ERROR = False

with open("data/counterfactual_results.jsonl", "r", encoding="utf-8") as f:
    data_list = [json.loads(line) for line in f]

perturbed_option_list = []

for item in data_list:
    custom_id = item['custom_id']
    response = item['response']
    body = response['body']
    error = body['error']
    if error is not None:
        HAS_ERROR = True
        print(textwrap.dedent(f"""
            ----------------
            custom_id: {custom_id}
            error: {error}
            ----------------
        """).strip())
    else:
        output = body['output']
        for out in output:
            if out["type"] == "message":
                content = out["content"][0]
                print(custom_id+"\n")
                text = json.loads(content["text"])
                perturbed_option = text["perturbed_option"]
                perturbed_option_list.append(
                    {'custom_id': custom_id, 'perturbed_option': perturbed_option})

# 对perturbed_option_list中的字典元素按字典里的custom_id升序排序
perturbed_option_list.sort(key=lambda x: int(x['custom_id']))

with open("data/qwen3_logiqa_results_answers.jsonl", "r", encoding="utf-8") as f:
    qwen3_logiqa_results_answers = [json.loads(line) for line in f]

with open("data/perturbed_option_list.jsonl", "w", encoding="utf-8") as f:
    for i, (perturbed_option, item) in enumerate(zip(perturbed_option_list, qwen3_logiqa_results_answers)):
        if int(perturbed_option['custom_id']) != item['id']:
            print(
                f"id不匹配:\nperturbed_option:{perturbed_option['custom_id']}\nitem:{item['id']}\n")
            break
        item['perturbed_option'] = perturbed_option['perturbed_option']
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
