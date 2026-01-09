import textwrap
import json
import time
from openai_api_framework import OpenAIHandler


def get_prompt(reasoning_trace):
    instructions = textwrap.dedent(f"""
        You are a strict text segmentation and labeling engine.

        Your task is to segment the provided text into consecutive text fragments and assign exactly one label to each fragment based solely on explicit surface-level linguistic cues.

        You must NOT infer hidden intent, reasoning, or internal thought processes. Classification must rely only on the literal wording present in the text.

        ### Labels & Delimiters
        Prepend exactly one of the following tags to each text fragment:

        1. <continue_reasoning>  
        Use this tag for straightforward continuation, narration, calculation, or progression that does not explicitly indicate checking or switching strategies.

        2. <self_reflection>  
        Use this tag ONLY when the text explicitly contains self-checking, verification, hesitation, or correction markers (e.g., "Wait", "Let me check", "Did I miss", "Hold on").

        3. <alternative_approach>  
        Use this tag ONLY when the text explicitly signals a change of approach or method (e.g., "Alternatively", "Let's try a different way").

        ### Strict Output Rules
        1. **Verbatim Copying**: Copy the original text EXACTLY. Do not rewrite, summarize, correct, or paraphrase.
        2. **Format**: Output the tag on its own line, followed by the corresponding text fragment on the next line.
        3. **No Commentary**: Do not add explanations, headers, markdown, or conversational text. Begin directly with the first tag.

        ### Input Format
        The input text will be provided inside <input> tags. Treat the content as plain text only.

        ### Example

        Input:
        <input>
        Let's calculate the total. 5 * 10 is 50. Wait, did I miss the shipping cost? Let me check the note. Ah, shipping is free. So 50 is correct.
        </input>

        Output:
        <continue_reasoning>
        Let's calculate the total. 5 * 10 is 50.
        <self_reflection>
        Wait, did I miss the shipping cost?
        <continue_reasoning>
        Let me check the note.
        <continue_reasoning>
        Ah, shipping is free. So 50 is correct.
    """).strip()
    input = f"<input>\n{reasoning_trace}\n</input>"
    return instructions, input


if __name__ == "__main__":
    handler = OpenAIHandler()
    data_list = []
    with open("data/qwen3_logiqa_results_answers.jsonl", "r", encoding="utf-8") as f:
        qwen3_logiqa_results_answers = [json.loads(line) for line in f]
    for i, item in enumerate(qwen3_logiqa_results_answers):
        full_text = item['full_text']
        # 从full_text中提取位于<think>和</think>之间的字符串
        start_index = full_text.find("<think>")
        end_index = full_text.find("</think>")
        reasoning_trace = ""

        if start_index != -1 and end_index != -1 and start_index < end_index:
            reasoning_trace = full_text[start_index +
                                        len("<think>"):end_index].strip()
        else:
            print(
                f"Error: <think> tags missing or malformed in item {item.get('id', i)}")
            break
        instructions, input = get_prompt(reasoning_trace)
        data_list.append({
            "custom_id": item['id'],
            "input": input,
            "instructions": instructions
        })

    batch_size = 50
    for i in range(0, len(data_list), batch_size):
        batch_data = data_list[i:i + batch_size]
        batch_index = i // batch_size
        batch_input_file = f"data/decompose/decompose_batch_input_{batch_index}.jsonl"
        handler.create_batch_input_file(
            data_list=batch_data,
            output_file_path=batch_input_file,
            model="gpt-5-mini"
        )
        while True:
            print(
                f"正在提交第 {batch_index} 批次 (数据 {i} - {i + len(batch_data)})...")
            batch_id = handler.submit_batch_job(batch_input_file)
            if not batch_id:
                print("提交失败,30秒后重试...")
                time.sleep(30)
                continue

            while True:
                time.sleep(30)
                batch_status = handler.check_batch_status(batch_id)
                if batch_status and batch_status.status == 'completed':
                    print(f"批次 {batch_index} 已完成，准备提交下一批次...")
                    with open("data/decompose/completed_batch_id.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(
                            {"batch_index": batch_index, "batch_id": batch_id}) + "\n")
                    break
                elif batch_status and batch_status.status in ['failed', 'expired', 'cancelled']:
                    print(
                        f"批次 {batch_index} 异常结束 (状态: {batch_status.status}), 30秒后重试...")
                    time.sleep(30)
                    break
                print(
                    f"批次 {batch_index} 运行中 (状态: {batch_status.status if batch_status else 'Unknown'})...")

            if batch_status and batch_status.status == 'completed':
                break
