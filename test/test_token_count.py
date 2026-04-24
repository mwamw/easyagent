import json
from core.history import CanonicalMessage, CanonicalBlock
from context.token.counter import TokenCounter
from core.providers import create_codec
from context.compressor.history import BaseHistoryCompactor, LLMHistoryCompactor

def create_test_history():
    history = []
    
    # 1. 简单的文本消息
    history.append(CanonicalMessage(
        role="user", 
        content=[CanonicalBlock(type="text", text="测试 Token 计数。")]
    ))
    
    # 2. 包含巨大参数的工具调用
    huge_args = '{"query": "搜索", "data": "' + ("A" * 1000) + '"}'
    history.append(CanonicalMessage(
        role="assistant",
        content=[
            CanonicalBlock(type="function_call", name="test_tool", arguments=huge_args)
        ]
    ))
    
    # 3. 包含巨大返回值的工具结果
    huge_output = "这里是工具的极长返回结果：" + ("B" * 2000)
    history.append(CanonicalMessage(
        role="tool",
        content=[
            CanonicalBlock(type="function_response", call_id="call_1", name="test_tool", output=huge_output)
        ]
    ))
    
    return history

def main():
    print("="*50)
    print("开始测试 Token 统计准确性")
    print("="*50)

    # 1. 初始化计数器和 Codec
    counter = TokenCounter(model="gpt-4")
    codec = create_codec("openai")
    
    # 2. 构造测试数据
    canonical_history = create_test_history()
    
    print("\n--- 1. 直接统计纯文本 (作为基准参照) ---")
    raw_text = json.dumps([msg.to_dict() for msg in canonical_history], ensure_ascii=False)
    raw_text_tokens = counter.count(raw_text)
    print(f"将历史记录直接转为纯 JSON 字符串的 Token 数: {raw_text_tokens}")
    
    print("\n--- 2. 使用 Codec 标准统计 ---")
    # 转为特定模型格式
    replay_history = codec.canonical_to_replay(canonical_history)
    # 计算完整的 Request Tokens (包含 message 结构损耗)
    codec_tokens = codec.count_request_tokens(counter, replay_history)
    print(f"Codec 针对 OpenAI API 格式计算的准确 Token 数: {codec_tokens}")
    
    print("\n--- 3. 历史压缩器内部的 _messages_token_count 统计 ---")
    # 创建一个哑的压缩器只是为了测 token
    from unittest.mock import MagicMock
    dummy_llm = MagicMock()
    compactor = LLMHistoryCompactor(llm=dummy_llm, token_counter=counter)
    
    compactor_tokens = compactor._messages_token_count(canonical_history)
    print(f"Compactor 内部计算的历史 Token 数: {compactor_tokens}")
    
    print("\n--- 4. 结论 ---")
    if codec_tokens > 500 and compactor_tokens > 500:
        print("✅ 统计成功！能够正确识别出内部巨型工具调用的 Token 数，没有忽略 JSON payload。")
    else:
        print("❌ 统计失败！Token 数过小，说明工具调用参数和返回值可能被忽略了。")
        
    print(f"比例差异: Codec({codec_tokens}) vs Compactor({compactor_tokens})")

if __name__ == "__main__":
    main()
