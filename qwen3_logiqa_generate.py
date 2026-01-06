import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
from datasets import load_dataset, Dataset
import textwrap

OUTPUT_FILE = "qwen3_logiqa_results.jsonl"


def load_LogiQA():
    return load_dataset(
        "lucasmccabe/logiqa",
        revision="refs/convert/parquet",
        split="train"
    )


def format_prompt(item):
    context = item['context']
    query = item['query']
    options = item['options']

    system_prompt = textwrap.dedent("""
        You are a logical reasoning assistant. 
        You must think step-by-step before answering.
    """).strip()

    user_content = textwrap.dedent(f"""
        Context:
        {context}
        
        Question: 
        {query}
        
        Select the single best answer from the options below:
        A){options[0]}
        B){options[1]}
        C){options[2]}
        D){options[3]}
    """).strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    # LogiQA dataset uses 'label' (0-3), map to A-D for consistency
    label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    return messages, label_map.get(item['correct_option'], "")


def generate_with_qwen3():
    MODEL_ID = "Qwen/Qwen3-8b"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    dataset = load_LogiQA()
    # 显式断言类型以消除 IDE 关于 len() 的类型警告
    assert isinstance(dataset, Dataset)

    # Use 'w' to overwrite or 'a' to append. Open once for efficiency.
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # 遍历所有数据
        for i, item in enumerate(tqdm(dataset, total=len(dataset), desc="推理进度")):
            # 1. 获取格式化后的消息列表和正确答案
            messages, correct_label = format_prompt(item)

            # 2. 应用 Chat Template
            # 这会将 messages 列表转换为模型原生的字符串格式 (例如包含 <|im_start|> 等 tag)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 3. 转换为 Tensor 并移动到模型所在的设备
            model_inputs = tokenizer([text], return_tensors="pt").to(
                model.device)

            # 4. 模型生成
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

            # 5. 解码输出
            full_sequence_ids = generated_ids[0].tolist()
            full_sequence_text = tokenizer.decode(
                generated_ids[0], skip_special_tokens=False)
            result_data = {
                "id": i,
                # [核心字段] 完整的 Token IDs，直接喂给模型 forward() 即可提取激活值，无歧义
                "full_ids": full_sequence_ids,
                # [辅助字段] 包含特殊字符的完整文本，用于人工检查
                "full_text": full_sequence_text,
                # 记录正确答案以便后续对比
                "label": correct_label
            }

            f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    generate_with_qwen3()
