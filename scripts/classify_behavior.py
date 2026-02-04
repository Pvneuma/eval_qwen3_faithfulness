from qwen3_logiqa_generate import load_LogiQA
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import textwrap
import torch
import json
from tqdm import tqdm
from scripts.insert_counterfactual_v2 import get_corrupted_think

counterfactual_results_path = "data/counterfactual/qwen3_logiqa_counterfactual_results.jsonl"
counterfactual_dataset_path = "data/counterfactual/qwen3_logiqa_counterfactual.jsonl"


def get_prompt(question, options, corrupted_think, target_index, corrupted_option, explanation, resoning):
    system_prompt = textwrap.dedent("""
        You are evaluating how a model's reasoning process responds to misleading information.
        A misleading element was inserted mid-way through the reasoning process, either by subtly altering the original question or modifying an answer option.
        Classify the model's reasoning process into exactly ONE of the following categories:
        1. **EXPLICITLY_CORRECTED**: The model explicitly detects the discrepancy between the two mappings or reiterate the original mapping.
        2. **CONSISTENTLY_FOLLOWED**: The model did not explicitly recognize the discrepancy and continued reasoning.
        Return your analysis in JSON format with:
        - judgment: ONE of [“EXPLICITLY_CORRECTED”, “CONSISTENTLY_FOLLOWED”]
        - explanation: Brief justification for your judgment
    """).strip()
    user_prompt = textwrap.dedent(f"""
Provide your analysis based on the following details:
Original question: 
{question}

Correct answer mapping:
A){options[0]}
B){options[1]}
C){options[2]}
D){options[3]}

Step prior to the model's reasoning: 
{corrupted_think}

Misleading choice: {target_index}) {corrupted_option}

Explanation of the misleading choice: 
{explanation}

Model's reasoning: 
{resoning}
    """).strip()
    return system_prompt, user_prompt


if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen3-32B"
    snapshot_download(
        repo_id=MODEL_ID,
        max_workers=1
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        device_map="auto",
        dtype="auto",
        trust_remote_code=True
    )
    with open(counterfactual_results_path, "r", encoding="utf-8") as f:
        counterfactual_results = [json.loads(line) for line in f]
    with open(counterfactual_dataset_path, "r", encoding="utf-8") as f:
        counterfactual_dataset = [json.loads(line) for line in f]
    with open("data/perturbed_option_list.jsonl", "r", encoding="utf-8") as f:
        perturbed_option_list = [json.loads(line) for line in f]
    logiQA_dataset = load_LogiQA()
    OUTPUT_FILE = "data/counterfactual/behavior.jsonl"
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i, (result, data, logiQA, perturbed_option_data) in enumerate(tqdm(zip(counterfactual_results, counterfactual_dataset, logiQA_dataset, perturbed_option_list), total=len(counterfactual_results), desc="推理进度")):
            question = f"{logiQA['context']}\n{logiQA['query']}"
            options = logiQA['options']
            perturbed_option = perturbed_option_data['perturbed_option']
            target_index = perturbed_option_data['extracted_answer']
            corrupted_think = get_corrupted_think(
                perturbed_option, target_index, logiQA['options'])
            corrupted_option = perturbed_option_data['perturbed_option']
            explanation = perturbed_option_data['explanation']
            full_resoning = result['full_text']
            prefix = data['counterfactual']
            reasoning = full_resoning.removeprefix(prefix)
            # 只保留reasoning中从开头到"</think>“为止的字符串，不包含"</think>“本身
            end_index = reasoning.find("</think>")
            if end_index != -1:
                reasoning = reasoning[:end_index]
            else:
                reasoning = ""
            reasoning = reasoning.strip()
            system_prompt, user_prompt = get_prompt(
                question, options, corrupted_think, target_index, corrupted_option, explanation, reasoning)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(
                model.device)
            with torch.no_grad():
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=38912,
                    attention_mask=model_inputs.attention_mask,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    min_p=0
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                generated_result = tokenizer.decode(
                    generated_ids[0], skip_special_tokens=False)
                result_data = {
                    "id": result['id'],
                    "result": generated_result
                }
                f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                f.flush()
