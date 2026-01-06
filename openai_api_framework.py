import os
import yaml
import json
from typing import List, Dict, Optional, Any
from openai import OpenAI, OpenAIError


class OpenAIHandler:
    def __init__(self):
        """
        初始化 OpenAI 客户端。

        配置仅从 config.yml 读取。
        """
        api_key = None

        if os.path.exists("config.yml"):
            try:
                with open("config.yml", "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    if config:
                        # 尝试读取 openai_api_key 或 openai.api_key
                        api_key = config.get("openai_api_key")
            except Exception as e:
                print(f"读取 config.yml 失败: {e}")

        if not api_key:
            raise ValueError("未在 config.yml 中找到有效的 API Key")

        self.client = OpenAI(
            api_key=api_key,
        )

    def create_batch_input_file(self,
                                data_list: List[Dict[str, Any]],
                                output_file_path: str,
                                model: str = "gpt-5-mini"):
        """
        辅助方法：将数据列表转换为 Batch API 需要的 JSONL 格式。
        Endpoint: /v1/responses
        """
        with open(output_file_path, "w", encoding="utf-8") as f:
            for item in data_list:
                # 构造 /v1/responses 的请求体
                body = {
                    "model": model,
                    "input": item.get("input"),
                    "instructions": item.get("instructions", "")
                }

                batch_request = {
                    "custom_id": str(item.get("custom_id", "")),
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body
                }
                f.write(json.dumps(batch_request, ensure_ascii=False) + "\n")

    def submit_batch_job(self, jsonl_file_path: str) -> Optional[str]:
        """上传文件并提交 Batch 任务"""
        try:
            # 1. 上传文件
            with open(jsonl_file_path, "rb") as f:
                file_response = self.client.files.create(
                    file=f,
                    purpose="batch"
                )

            # 2. 创建 Batch 任务
            batch_response = self.client.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/responses",
                completion_window="24h"
            )
            return batch_response.id
        except OpenAIError as e:
            print(f"OpenAI API 请求错误: {e}")
            return None

    def check_batch_status(self, batch_id: str) -> Any:
        """查询 Batch 任务状态"""
        try:
            return self.client.batches.retrieve(batch_id)
        except Exception as e:
            print(f"查询状态失败: {e}")
            return None

    def retrieve_batch_results(self, output_file_id: str) -> Optional[str]:
        """下载 Batch 结果"""
        try:
            return self.client.files.content(output_file_id).text
        except Exception as e:
            print(f"下载结果失败: {e}")
            return None


if __name__ == "__main__":
    # 使用示例
    handler = OpenAIHandler()

    # 1. 准备数据
    test_data = [
        {"custom_id": "req-1", "input": "测试输入1", "instructions": "指令1"},
        {"custom_id": "req-2", "input": "测试输入2", "instructions": "指令2"}
    ]
    jsonl_file = "batch_input.jsonl"

    print("正在生成 Batch 文件...")
    handler.create_batch_input_file(test_data, jsonl_file)

    # 2. 提交任务
    print("正在提交 Batch 任务...")
    batch_id = handler.submit_batch_job(jsonl_file)

    if batch_id:
        print(f"Batch 任务已提交，ID: {batch_id}")
        print("请稍后使用 check_batch_status 查询状态。")
