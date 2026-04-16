from anthropic import Anthropic

client = Anthropic(
    api_key="aa",
    base_url="http://127.0.0.1:5124",
)

response = client.messages.create(
    model="qwen3.5-9b",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "你好，做个自我介绍"}
    ]
)

print(response.content[0])