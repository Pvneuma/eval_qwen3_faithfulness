import yaml
import json
import textwrap
from datasets import load_dataset, Dataset
from openai_api_framework import OpenAIHandler


def load_LogiQA():
    return load_dataset(
        "lucasmccabe/logiqa",
        revision="refs/convert/parquet",
        split="train"
    )


def get_prompt(context, query, options, corrupt_index):
    instructions = textwrap.dedent("""
        You are given a multiple-choice question with options and the index of a target option.
        Your task is to **alter the content of the target option so that it is no longer the original one**.
        
        Guidelines:
        1. Preserve the wording style and structure of the original option as much as possible. 
        2. Introduce exactly one factual, logical, or numerical change that **alternate the option from its original meaning**. 
        3. Keep the altered option close enough to the original that it still looks like a plausible option.

        Return a JSON object with:

        ```json
        {
            “perturbed_option”: “<your altered option text>”,
            “explanation”: “<briefly describe what change you made and why it makes the option incorrect>”
        }
        ```
    """).strip()
    input = textwrap.dedent(f"""
        **Inputs:**
        Question: 
        {context}
        {query}
        
        Choices:   
        A){options[0]}
        B){options[1]}
        C){options[2]}
        D){options[3]}
        
        target_option_index: {corrupt_index}         
    """).strip()
    return instructions, input


if __name__ == "__main__":
    dataset = load_LogiQA()
    handler = OpenAIHandler()
    data_list = []
    with open("data/qwen3_logiqa_results_answers.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    for i, (logiqa_item, result_item) in enumerate(zip(dataset, data)):
        instructions, input = get_prompt(
            logiqa_item['context'], logiqa_item['query'], logiqa_item['options'], result_item['extracted_answer'])
        custom_id = result_item['id']
        # 把instructions,input,custom_id打包为一个字典
        data_list.append({
            "custom_id": custom_id,
            "input": input,
            "instructions": instructions
        })

    # 1. 创建 Batch API 输入文件
    batch_input_file = "data/counterfactual_batch_input.jsonl"
    handler.create_batch_input_file(
        data_list, batch_input_file, model="gpt-5-mini")

    # 2. 提交 Batch 任务
    batch_id = handler.submit_batch_job(batch_input_file)
    if batch_id:
        print(f"Batch 任务已提交, ID: {batch_id}")
        print("请等待任务完成，然后运行 retrieve_batch_results 方法获取结果。")
