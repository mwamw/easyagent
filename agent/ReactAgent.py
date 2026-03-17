"""
ReAct Agent

实现显式的思考链（Thought → Action → Observation）模式。
与 BasicAgent 使用 Function Calling 不同，ReactAgent 通过
prompt 引导 LLM 输出结构化的推理过程。
"""
from typing_extensions import override
from typing import Optional, List, Dict, Any, Tuple
import re
import json
import logging

from .BasicAgent import BasicAgent
from core.llm import EasyLLM
from core.Message import Message, UserMessage, SystemMessage, AssistantMessage
from core.Config import Config
from Tool.ToolRegistry import ToolRegistry
from core.Exception import *

logger = logging.getLogger(__name__)


class ReactAgent(BasicAgent):
    """
    ReAct Agent - 显式思考链 Agent
    
    实现 Thought → Action → Observation 的推理循环。
    每一步推理过程都会被显式输出，便于理解和调试。
    
    与 BasicAgent 的区别：
    - BasicAgent: 使用 Function Calling，思考过程隐式
    - ReactAgent: 使用 Prompt 引导，思考过程显式
    
    Example:
        >>> agent = ReactAgent(
        ...     name="react",
        ...     llm=llm,
        ...     tool_registry=registry,
        ...     enable_tool=True
        ... )
        >>> result = agent.invoke("北京今天天气怎么样？")
        
        # 输出示例：
        # Thought: 用户想知道北京的天气，我需要调用天气查询工具
        # Action: get_weather
        # Action Input: {"city": "北京"}
        # Observation: 北京今天晴，气温 -2°C 到 8°C
        # Thought: 我已经获取到天气信息，可以回答用户了
        # Final Answer: 北京今天是晴天...
    """
    
    def __init__(
        self,
        name: str,
        llm: EasyLLM,
        system_prompt: Optional[str] = None,
        enable_tool: bool = True,
        tool_registry: Optional[ToolRegistry] = None,
        description: Optional[str] = None,
        config: Optional[Config] = None,
        verbose: bool = True,
    ):
        """
        初始化 ReAct Agent
        
        Args:
            name: Agent 名称
            llm: LLM 实例
            system_prompt: 额外的系统提示词
            enable_tool: 是否启用工具（ReAct 通常需要工具）
            tool_registry: 工具注册表
            description: Agent 描述
            config: 配置
            verbose: 是否输出详细的推理过程
        """
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            enable_tool=enable_tool,
            tool_registry=tool_registry,
            description=description,
            config=config
        )
        self.verbose = verbose
        self.scratchpad: List[str] = []  # 记录推理过程
    
    @override
    def invoke(self, query: str, max_iter: int = 10, temperature: float = 0.7, **kwargs) -> str:
        """
        执行 ReAct 推理循环
        
        Args:
            query: 用户输入
            max_iter: 最大迭代次数
            temperature: 温度参数
            
        Returns:
            最终答案
        """
        self._validate_invoke_params(query, max_iter, temperature)
        
        # 清空 scratchpad
        self.scratchpad = []
        
        if not self.enable_tool or not self.tool_registry:
            # 没有工具，直接回答
            return self._direct_answer(query, temperature, **kwargs)
        
        logger.info(f"开始 ReAct 推理: {query[:50]}...")
        
        # 构建初始消息
        system_prompt = self._build_react_prompt()
        user_message = f"问题: {query}"
        
        iteration = 0
        final_answer = None
        
        while iteration < max_iter:
            iteration += 1
            logger.debug(f"ReAct 迭代 {iteration}")
            
            # 构建当前消息
            messages = [
                SystemMessage(system_prompt),
                UserMessage(user_message + "\n\n" + self._get_scratchpad())
            ]
            
            # 调用 LLM
            try:
                response = self.llm.invoke(messages, temperature=temperature, **kwargs)
            except Exception as e:
                logger.error(f"LLM 调用失败: {e}")
                final_answer = f"推理过程中发生错误: {e}"
                break
            
            if self.verbose:
                print(f"\n{response}")
            
            # 解析响应
            parsed = self._parse_response(response)
            
            if parsed["type"] == "final_answer":
                final_answer = parsed["content"]
                self.scratchpad.append(f"Thought: {parsed.get('thought', '我现在知道答案了')}")
                self.scratchpad.append(f"Final Answer: {final_answer}")
                break
            
            elif parsed["type"] == "action":
                thought = parsed.get("thought", "")
                action = parsed["action"]
                action_input = parsed["action_input"]
                
                self.scratchpad.append(f"Thought: {thought}")
                self.scratchpad.append(f"Action: {action}")
                self.scratchpad.append(f"Action Input: {json.dumps(action_input, ensure_ascii=False)}")
                
                # 执行工具
                observation = self._execute_action(action, action_input)
                self.scratchpad.append(f"Observation: {observation}")
                
                if self.verbose:
                    print(f"Observation: {observation}")
            
            elif parsed["type"] == "thought_only":
                # 只有思考，没有行动
                self.scratchpad.append(f"Thought: {parsed['thought']}")
            
            else:
                # 无法解析，尝试作为最终答案
                logger.warning("无法解析响应，尝试作为最终答案")
                final_answer = response
                break
        
        if final_answer is None:
            final_answer = "超过最大迭代次数，未能完成推理"
            logger.warning(final_answer)
        
        # 保存历史
        self.add_message(UserMessage(query))
        self.add_message(AssistantMessage(final_answer))
        
        return final_answer
    
    def _build_react_prompt(self) -> str:
        """构建 ReAct 系统提示词"""
        tools_desc = self._format_tools_for_prompt()
        
        prompt = f"""你是一个智能助手，使用 ReAct（Reasoning + Acting）方法解决问题。

## 工作流程
你需要按照以下格式进行思考和行动：

Thought: 分析当前情况，思考下一步应该做什么
Action: 要使用的工具名称
Action Input: 工具的输入参数（JSON 格式）

工具执行后，你会看到：
Observation: 工具返回的结果

然后继续思考...

当你有足够信息回答问题时：
Thought: 我现在知道最终答案了
Final Answer: 对用户问题的完整回答

## 可用工具
{tools_desc}

## 注意事项
1. 每次只调用一个工具
2. 工具名称必须完全匹配
3. Action Input 必须是有效的 JSON
4. 仔细分析 Observation 再决定下一步
5. 如果不需要工具，直接给出 Final Answer

{self.system_prompt or ''}
"""
        return prompt
    
    def _format_tools_for_prompt(self) -> str:
        """格式化工具描述"""
        if not self.tool_registry:
            return "（无可用工具）"
        
        tools = []
        for name, tool in self.tool_registry.tools.items():
            schema = tool.parameters.model_json_schema()
            params_desc = json.dumps(schema.get("properties", {}), ensure_ascii=False, indent=2)
            tools.append(f"- {name}: {tool.description}\n  参数: {params_desc}")
        
        return "\n".join(tools) if tools else "（无可用工具）"
    
    def _get_scratchpad(self) -> str:
        """获取当前 scratchpad 内容"""
        if not self.scratchpad:
            return ""
        return "\n".join(self.scratchpad)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析 LLM 响应
        
        Returns:
            {
                "type": "action" | "final_answer" | "thought_only" | "unknown",
                "thought": str,
                "action": str,
                "action_input": dict,
                "content": str
            }
        """
        result = {"type": "unknown"}
        
        # 提取 Final Answer
        final_match = re.search(r'Final Answer[:\s]*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        if final_match:
            # 检查是否在 Final Answer 之前有完整内容
            final_content = final_match.group(1).strip()
            # 获取 Final Answer 之后的所有内容
            final_idx = response.lower().find('final answer')
            remaining = response[final_idx:].split('\n', 1)
            if len(remaining) > 1:
                final_content = remaining[0].replace('Final Answer:', '').replace('Final Answer', '').strip()
                if remaining[1].strip():
                    final_content += '\n' + remaining[1].strip()
            
            result["type"] = "final_answer"
            result["content"] = final_content
            
            # 尝试提取 thought
            thought_match = re.search(r'Thought[:\s]*(.+?)(?=Final Answer|Action|\n\n|$)', response, re.IGNORECASE | re.DOTALL)
            if thought_match:
                result["thought"] = thought_match.group(1).strip()
            
            return result
        
        # 提取 Thought
        thought_match = re.search(r'Thought[:\s]*(.+?)(?=Action|\n\n|$)', response, re.IGNORECASE | re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        
        # 提取 Action
        action_match = re.search(r'Action[:\s]*([^\n]+)', response, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()
            
            # 提取 Action Input
            input_match = re.search(r'Action Input[:\s]*(.+?)(?=Observation|\n\n|$)', response, re.IGNORECASE | re.DOTALL)
            action_input = {}
            
            if input_match:
                input_str = input_match.group(1).strip()
                try:
                    # 尝试解析 JSON
                    action_input = json.loads(input_str)
                except json.JSONDecodeError:
                    # 尝试从代码块中提取
                    json_match = re.search(r'```(?:json)?\s*(.+?)\s*```', input_str, re.DOTALL)
                    if json_match:
                        try:
                            action_input = json.loads(json_match.group(1))
                        except:
                            action_input = {"input": input_str}
                    else:
                        action_input = {"input": input_str}
            
            result["type"] = "action"
            result["thought"] = thought
            result["action"] = action
            result["action_input"] = action_input
            return result
        
        # 只有 Thought
        if thought:
            result["type"] = "thought_only"
            result["thought"] = thought
            return result
        
        return result
    
    def _execute_action(self, action: str, action_input: Dict[str, Any]) -> str:
        """执行工具动作"""
        if not self.tool_registry:
            return "错误：没有可用的工具注册表"
        
        try:
            # 查找工具
            if action not in self.tool_registry.tools:
                # 尝试模糊匹配
                for tool_name in self.tool_registry.tools:
                    if tool_name.lower() == action.lower():
                        action = tool_name
                        break
                else:
                    return f"错误：未找到工具 '{action}'。可用工具: {list(self.tool_registry.tools.keys())}"
            
            result = self.tool_registry.executeTool(action, action_input)
            return str(result) if result else "工具执行完成，无返回结果"
        
        except Exception as e:
            logger.error(f"工具执行失败: {e}")
            return f"工具执行错误: {e}"
    
    def _direct_answer(self, query: str, temperature: float, **kwargs) -> str:
        """不使用工具直接回答"""
        messages = [
            SystemMessage(self.system_prompt or "你是一个有用的助手。"),
            UserMessage(query)
        ]
        
        try:
            response = self.llm.invoke(messages, temperature=temperature, **kwargs)
            self.add_message(UserMessage(query))
            self.add_message(AssistantMessage(response))
            return response
        except Exception as e:
            raise LLMInvokeError(f"LLM 调用失败: {e}") from e
    
    def get_scratchpad(self) -> List[str]:
        """获取推理过程记录"""
        return self.scratchpad.copy()
    
    def get_reasoning_trace(self) -> str:
        """获取格式化的推理轨迹"""
        return "\n".join(self.scratchpad)
    
    @override
    def get_enhanced_prompt(self) -> str:
        """获取增强提示词，包含记忆上下文和 ReAct 指令"""
        react_prompt = self._build_react_prompt()
        # 注入记忆系统提示和 Working Memory 便签本
        react_prompt += self._build_memory_prompt()
        return react_prompt
