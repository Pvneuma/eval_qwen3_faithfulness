from openai_api_framework import OpenAIHandler
import json

handler = OpenAIHandler()
output_file_path = "data/decompose/output/decompose_results.jsonl"
batch_id_file_path = "data/decompose/completed_batch_id.jsonl"


with open(batch_id_file_path, "r", encoding="utf-8") as f:
    completed_batch_id_list = [json.loads(line)['batch_id'] for line in f]

for batch_id in completed_batch_id_list:
    batch = handler.check_batch_status(batch_id)
    if batch.request_counts.failed > 0:
        print(f"批次 {batch_id} 有处理失败 (失败数: {batch.request_counts.failed}), 程序退出")
        exit(1)

handler.retrieve_batch_batch_results(output_file_path, batch_id_file_path)
