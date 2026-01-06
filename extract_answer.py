import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
from datasets import load_dataset
import textwrap

INPUT_FILE = "qwen3_logiqa_results.jsonl"
OUTPUT_FILE = "qwen3_logiqa_results_answers.jsonl"


def format_prompt(item):
    full_text = item['full_text']
    # 从response中提取位于"</think>"后的全部字符串
    start_index = full_text.find("</think>")
    if start_index != -1:
        response = full_text[start_index + len("</think>"):]
    else:
        response = full_text

    prompt = textwrap.dedent(f"""
        You will be shown a response to a question. Your task is to extract the final selected option. Output only the corresponding letter (e.g., C). Output plain text only. Do NOT use Markdown under any circumstances.
        
        The response:
        
        {response}
    """).strip()
    messages = [
        {"role": "user", "content": prompt}
    ]
    return messages


def generate_with_qwen3():
    MODEL_ID = "Qwen/Qwen3-8b"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Use 'w' to overwrite or 'a' to append. Open once for efficiency.
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # 遍历所有数据
        for i, item in enumerate(tqdm(dataset, total=len(dataset), desc="推理进度")):
            # 1. 获取格式化后的消息列表和正确答案
            messages = format_prompt(item)

            # 2. 应用 Chat Template
            # 这会将 messages 列表转换为模型原生的字符串格式 (例如包含 <|im_start|> 等 tag)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # 3. 转换为 Tensor 并移动到模型所在的设备
            model_inputs = tokenizer([text], return_tensors="pt").to(
                model.device)

            # 4. 模型生成
            with torch.no_grad():
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=4096,
                    attention_mask=model_inputs.attention_mask,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    min_p=0
                )

            # 5. 解码输出
            full_sequence_text = tokenizer.decode(
                generated_ids[0], skip_special_tokens=True)
            # 把full_sequence_text中的最后一个字符赋值给extracted_answer，并且检查是否属于A，B，C，D中的一个（大小写不敏感），如果不属于，将当前索引i和full_sequence_text写入当前目录下的extract_answer_log.jsonl
            extracted_answer = full_sequence_text[-1]
            if extracted_answer.upper() not in ['A', 'B', 'C', 'D']:
                with open("extract_answer_log.jsonl", "a", encoding="utf-8") as log_f:
                    log_f.write(json.dumps({"id": i, "full_sequence_text": full_sequence_text}, ensure_ascii=False) + "\n")
                extracted_answer = "" # Set to empty if invalid

            item["extracted_answer"] = extracted_answer
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    generate_with_qwen3()
