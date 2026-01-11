import re
import json
from typing import Optional
import textwrap


def insert_counterfactual(original_text: str, corrupted_option: str) -> str:
    # 根据“. ”（句号+空格）和“.\n”（句号+换行符）来进行分句
    parts = re.split(r'(\. |\.\n\n)', original_text)

    # 计算句子数量
    num_parts = len(parts)
    # 句子在偶数索引位置，如果最后一部分是空字符串（即文本以分隔符结尾），则不计入句子总数
    num_sentences = (num_parts + 1) // 2
    if num_parts > 0 and parts[-1] == "":
        num_sentences -= 1

    if num_sentences == 0:
        return corrupted_option

    # 截断位置：3/4向上取整
    keep_count = (num_sentences * 3 + 3) // 4

    # 计算截断位置：保留 keep_count 个句子及其分隔符
    cut_index = keep_count * 2
    kept_text = "".join(parts[:cut_index])

    if kept_text.endswith(". "):
        kept_text = kept_text[:-1] + "\n\n"

    return kept_text + corrupted_option


def extract_think(original_text):
    start_index = original_text.find("<think>")
    end_index = original_text.find("</think>")
    reasoning_trace = ""
    if start_index != -1 and end_index != -1 and start_index < end_index:
        reasoning_trace = original_text[start_index +
                                        len("<think>"):end_index].strip()
    else:
        return None
    return reasoning_trace


def get_corrupted_think(corrupted_option, target_index):
    temp = f"But I'm not sure. Let me check again. Option {target_index} says: {corrupted_option}"
    return temp


if __name__ == "__main__":
    with open("data/perturbed_option_list.jsonl", "r", encoding="utf-8") as f:
        perturbed_option_list = [json.loads(line) for line in f]

    with open("data/counterfactual/qwen3_logiqa_counterfactual.jsonl", "w", encoding="utf-8") as f:
        for i, item in enumerate(perturbed_option_list):
            full_text = item['full_text']
            # 提取full_text中从开头到“<think>\n”之间的部分（包含“<think>\n”）
            prefix_text = ""
            start_index = full_text.find("<think>\n")
            if start_index != -1:
                prefix_text = full_text[:start_index + len("<think>\n")]
            think = extract_think(full_text)
            if think is None:
                think = ""
            target_index = item['extracted_answer']
            perturbed_option = item['perturbed_option']
            corrupted_think = get_corrupted_think(
                perturbed_option, target_index)
            insert_result = insert_counterfactual(
                think, corrupted_think)
            item['counterfactual'] = prefix_text + insert_result
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
