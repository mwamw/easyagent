# %% [markdown]
# # 测试 History Compactor (规则压缩、LLM压缩、混合压缩)
# 在 VSCode 中，你可以直接点击单元格上方的 "Run Cell" 来执行这段代码，它的体验和 .ipynb 笔记本完全一样！

# %%
import sys
import os
# 将项目根目录加入 path，以便导入 core 等模块
sys.path.append(os.path.abspath(".."))

import json
from core.llm import EasyLLM
from core.history import CanonicalMessage, CanonicalBlock
from context.token.counter import TokenCounter
from context.compressor.history import (
    RuleBasedHistoryCompactor,
    LLMHistoryCompactor,
    HybridHistoryCompactor
)

# 1. 配置 LLM
# TODO: 请填入你真实的测试模型配置，如果你的环境变量里已经有配置，直接写 provider 和 model 即可。
llm = EasyLLM(
    provider="openai",
    base_url="http://127.0.0.1:5124/v1",
    api_key="122",
    model="qwen3.5-9b",
)

token_counter = TokenCounter()

# %% [markdown]
# ### 2. 构造冗长的假数据历史 (Mock History)
# 模拟一个超长对话：包含多个工具调用和极长的工具输出

# %%
def create_mock_history():
    history = []
    
    # 轮次 1
    history.append(CanonicalMessage(
        role="user", 
        content=[CanonicalBlock(type="text", text="帮我搜索一下本地的文档，找一下关于EasyAgent的架构说明。")]
    ))
    history.append(CanonicalMessage(
        role="assistant",
        content=[
            CanonicalBlock(type="text", text="好的，我先用 grep_search 工具搜索一下相关的文档。"),
            CanonicalBlock(type="function_call", name="grep_search", arguments='{"query": "架构", "path": "./docs"}')
        ]
    ))
    history.append(CanonicalMessage(
        role="tool",
        content=[
            CanonicalBlock(type="function_response", call_id="call_1", name="grep_search", output="匹配到 1000 个结果：\n" + ("非常冗长的架构设计文档内容... \n" * 500))
        ]
    ))
    
    # 轮次 2
    history.append(CanonicalMessage(
        role="user", 
        content=[CanonicalBlock(type="text", text="帮我看看 manager.py 是怎么写的。")]
    ))
    history.append(CanonicalMessage(
        role="assistant",
        content=[
            CanonicalBlock(type="text", text="这就去查看源码。"),
            CanonicalBlock(type="function_call", name="view_file", arguments='{"path": "manager.py"}')
        ]
    ))
    history.append(CanonicalMessage(
        role="tool",
        content=[
            CanonicalBlock(type="function_response", call_id="call_2", name="view_file", output="class ContextManager:\n" + ("    def some_method(self):\n        pass\n" * 400))
        ]
    ))
    
    # 轮次 3 (最近一轮，应该被保护)
    history.append(CanonicalMessage(
        role="user", 
        content=[CanonicalBlock(type="text", text="根据刚才看到的源码，写一个总结。")]
    ))
    
    return [msg.to_dict() for msg in history]

history = create_mock_history()
print(f"原始历史消息总数: {len(history)}")
print(f"原始历史字数 (近似 Token): {len(json.dumps(history, ensure_ascii=False))}")

# %% [markdown]
# ### 3. 测试纯规则压缩 (RuleBasedHistoryCompactor)
# 预期：瞬间完成，旧工具的长输出被截断为最多 100 字符的流水账，而最近一轮的 "根据刚才看到的..." 原样保留。

# %%
rule_compactor = RuleBasedHistoryCompactor(token_counter=token_counter, recent_turns=1)

print("开始纯规则压缩...")
rule_result = rule_compactor.compact(history, max_tokens=10000)

print(f"\\n规则压缩完成！结果消息数: {len(rule_result)}")
print(f"结果字数 (近似 Token): {len(json.dumps(rule_result, ensure_ascii=False))}")
print("\\n【规则压缩后的前几条消息】 (注意查看它是如何被截断成骨架摘要的)：")
print(json.dumps(rule_result, indent=2, ensure_ascii=False))
print("\\n运行信息:", json.dumps(rule_compactor.get_last_run_info(), indent=2))

# %% [markdown]
# ### 4. 测试混合压缩 (HybridHistoryCompactor)
# 预期：
# - 场景 A (预算充足): rule 压缩完发现足够小，直接返回，不调 LLM。
# - 场景 B (预算紧张): rule 压缩完依然超标，触发大模型进行终极脱水。

# %%
hybrid_compactor = HybridHistoryCompactor(
    llm=llm, 
    token_counter=token_counter, 
    recent_turns=1
)

print("===== 场景 A: 预算充足 (10000 tokens) =====")
hybrid_result_A = hybrid_compactor.compact(history, max_tokens=10000)
print(f"\\n混合压缩完成 (A)！结果消息数: {len(hybrid_result_A)}")
print("运行信息:", json.dumps(hybrid_compactor.get_last_run_info(), indent=2))

print("\\n===== 场景 B: 预算紧张 (200 tokens) =====")
# 故意把 max_tokens 设得很小，逼迫它使用大模型
hybrid_result_B = hybrid_compactor.compact(history, max_tokens=200)
print(f"\\n混合压缩完成 (B)！结果消息数: {len(hybrid_result_B)}")
print("运行信息:", json.dumps(hybrid_compactor.get_last_run_info(), indent=2))
print("\\n【大模型深度脱水后的终极摘要】：")
print(json.dumps(hybrid_result_B, indent=2, ensure_ascii=False))
