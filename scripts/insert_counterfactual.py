import re
import json
from typing import Optional
import textwrap


def insert_counterfactual(decomposed_trace: str, original_text: str, corrupted_option: str) -> Optional[str]:
    """
    根据思维块规则截断 reasoning trace 并插入 corrupted_option。

    规则：
    1. 思维块定义：
       - 由 BACKTRACK step (<self_reflection> 或 <alternative_approach>) 开始，
         跟随任意数量的 CONTINUE steps (<continue_reasoning>)。
       - 特例：如果 trace 以 CONTINUE steps 开始，则这些 steps 组成第一个思维块。
    2. 定位中间思维块：
       - 索引 = (总块数 - 1) // 2
    3. 截断并插入：
       - 保留从开头到中间思维块结束的内容。
       - 此时需注意：保留的内容应与 original_text 的格式（换行符等）保持一致。
       - 追加 corrupted_option。
       - 丢弃之后的内容。
    """

    # 定义标签
    TAG_CONTINUE = "<continue_reasoning>"
    TAG_REFLECTION = "<self_reflection>"
    TAG_ALTERNATIVE = "<alternative_approach>"

    BACKTRACK_TAGS = {TAG_REFLECTION, TAG_ALTERNATIVE}
    ALL_TAGS = {TAG_CONTINUE, TAG_REFLECTION, TAG_ALTERNATIVE}

    # 使用正则拆分，保留标签
    # 模式匹配任何一个标签
    pattern = r'(<(?:continue_reasoning|self_reflection|alternative_approach)>)'
    parts = re.split(pattern, decomposed_trace)

    # 解析步骤 (Tag, Content)
    steps = []

    # parts 的结构通常是 [pre_text, tag, content, tag, content, ...]
    # 找到第一个标签的位置
    start_idx = -1
    for i, part in enumerate(parts):
        if part in ALL_TAGS:
            start_idx = i
            break

    if start_idx == -1:
        # 如果没有找到标签，直接返回原始内容 + corrupted_option
        return original_text + f"\n{corrupted_option}"

    # 从第一个标签开始遍历
    for i in range(start_idx, len(parts) - 1, 2):
        tag = parts[i]
        content = parts[i+1]
        steps.append({"tag": tag, "content": content})

    if not steps:
        return original_text + f"\n{corrupted_option}"

    # 构建思维块
    blocks = []
    current_block = []

    for step in steps:
        tag = step["tag"]
        is_backtrack = tag in BACKTRACK_TAGS

        if is_backtrack:
            # 如果遇到 BACKTRACK 标签，且当前块不为空，则当前块结束
            if current_block:
                blocks.append(current_block)
                current_block = []
            # BACKTRACK 标签开始一个新的块
            current_block.append(step)
        else:
            # CONTINUE 标签加入当前块
            current_block.append(step)

    # 加入最后一个块
    if current_block:
        blocks.append(current_block)

    # 计算目标块索引
    num_blocks = len(blocks)
    if num_blocks == 0:
        return corrupted_option

    target_index = (num_blocks - 1) // 2

    # 收集需要保留的文本内容（不含标签）
    kept_content_parts = []

    # 保留第一个标签前的内容（如果有）
    if start_idx > 0:
        kept_content_parts.append(parts[0])

    # 保留直到目标索引的块
    for i in range(target_index + 1):
        block = blocks[i]
        for step in block:
            kept_content_parts.append(step["content"])

    kept_raw_text = "".join(kept_content_parts)

    # 将 kept_raw_text 映射回 original_text 以保留原始格式
    # 提取 kept_raw_text 中的非空白字符作为匹配目标
    target_chars = [c for c in kept_raw_text if not c.isspace()]

    if not target_chars:
        return corrupted_option

    final_text_chars = []
    match_idx = 0
    target_len = len(target_chars)

    for char in original_text:
        if match_idx >= target_len:
            break
        final_text_chars.append(char)
        if not char.isspace():
            if char != target_chars[match_idx]:
                return None
            match_idx += 1

    if match_idx < target_len:
        return None

    result_text = "".join(final_text_chars)

    # 插入 corrupted_option
    result_text += f"\n\n{corrupted_option}"

    return result_text


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
    temp = f"But I'm not sure. Let me check again. Option {target_index} is: {corrupted_option}"
    return temp


if __name__ == "__main__":
    with open("data/decompose/output/decompose_results.jsonl", "r", encoding="utf-8") as f:
        decompose_results = [json.loads(line) for line in f]
    with open("data/perturbed_option_list.jsonl", "r", encoding="utf-8") as f:
        perturbed_option_list = [json.loads(line) for line in f]

    decompose_list = []
    for item in decompose_results:
        custom_id = item['custom_id']
        response = item['response']
        body = response['body']
        output = body['output']
        for out in output:
            if out["type"] == "message":
                content = out["content"][0]
                text = content["text"]
                decompose_list.append(
                    {'custom_id': custom_id, 'decomposed_trace': text})

    decompose_list.sort(key=lambda x: int(x['custom_id']))

    for i, (decompose, item) in enumerate(zip(decompose_list, perturbed_option_list)):
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
        corrupted_think = get_corrupted_think(perturbed_option, target_index)
        insert_result = insert_counterfactual(
            decompose['decomposed_trace'], think, corrupted_think)
        if insert_result is not None:
            with open("data/counterfactual/qwen3_logiqa_counterfactual.jsonl", "w", encoding="utf-8") as f:
                item['counterfactual'] = prefix_text + insert_result
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            print(f"id{item['id']}的文本不一致\n")
            break
