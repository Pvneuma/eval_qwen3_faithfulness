from openai_api_framework import OpenAIHandler


handler = OpenAIHandler()
handler.retrieve_batch_results("data/counterfactual_results.jsonl")