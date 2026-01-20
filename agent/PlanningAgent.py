"""
规划执行 Agent

先分析任务并制定计划，然后逐步执行每个步骤。
"""
from typing import Optional, List, Dict, Any
from typing_extensions import override
import json
import logging

from .BasicAgent import BasicAgent
from core.llm import EasyLLM
from core.Message import Message, UserMessage, SystemMessage, AssistantMessage
from core.Config import Config
from Tool.ToolRegistry import ToolRegistry
from core.Exception import *
from output.json_parser import JsonOutputParser
logger = logging.getLogger(__name__)


class PlanningAgent(BasicAgent):
    """
    规划执行 Agent
    
    先将复杂任务分解为子任务，然后逐步执行每个子任务。
    适合处理需要多步骤完成的复杂任务。
    
    工作流程：
    1. 分析任务，生成执行计划
    2. 逐步执行计划中的每个步骤
    3. 根据执行结果，可能进行计划调整
    4. 汇总结果，给出最终答案
    
    Example:
        >>> agent = PlanningAgent(
        ...     name="planner",
        ...     llm=llm,
        ...     tool_registry=registry,
        ...     enable_tool=True
        ... )
        >>> result = agent.invoke("帮我写一篇关于人工智能的调研报告")
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
        max_steps: int = 10,
        allow_replan: bool = True,
    ):
        """
        初始化规划 Agent
        
        Args:
            name: Agent 名称
            llm: LLM 实例
            system_prompt: 系统提示词
            enable_tool: 是否启用工具
            tool_registry: 工具注册表
            description: Agent 描述
            config: 配置
            max_steps: 最大执行步骤数
            allow_replan: 是否允许重新规划
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
        self.max_steps = max_steps
        self.allow_replan = allow_replan
        self.current_plan: List[str] = []
        self.execution_log: List[Dict[str, Any]] = []
        self.json_parser=JsonOutputParser()
    
    @override
    def invoke(self, query: str, max_iter: int = 10, temperature: float = 0.7, **kwargs) -> str:
        """执行规划任务"""
        self._validate_invoke_params(query, max_iter, temperature)
        
        logger.info(f"开始规划任务: {query[:50]}...")
        
        # 1. 生成计划
        plan = self._generate_plan(query, temperature)
        self.current_plan = plan
        logger.info(f"生成计划，共 {len(plan)} 个步骤")
        
        # 2. 执行计划
        results = []
        for i, step in enumerate(plan):
            if i >= self.max_steps:
                logger.warning("达到最大步骤数限制")
                break
            
            logger.info(f"执行步骤 {i+1}/{len(plan)}: {step[:50]}...")
            result = self._execute_step(step, temperature)
            results.append({
                "step": i + 1,
                "task": step,
                "result": result
            })
            self.execution_log.append(results[-1])
            
            # 检查是否需要重新规划
            if self.allow_replan and self._should_replan(result):
                logger.info("检测到需要重新规划")
                remaining_plan = self._replan(query, results, plan[i+1:], temperature)
                plan = plan[:i+1] + remaining_plan
                self.current_plan = plan
        
        # 3. 汇总结果
        final_answer = self._summarize_results(query, results, temperature)
        
        # 保存历史
        self.history.append(UserMessage(query))
        self.history.append(AssistantMessage(final_answer))
        
        return final_answer
    
    def _generate_plan(self, query: str, temperature: float) -> List[str]:
        """生成执行计划"""
        tools_desc = ""
        if self.enable_tool and self.tool_registry:
            tools_desc = f"\n可用工具：\n{self.tool_registry.get_tools_description()}"
        
        plan_prompt = f"""请分析以下任务，并将其分解为具体的执行步骤。

任务：{query}
{tools_desc}

请以 JSON 数组格式返回步骤列表，每个步骤是一个字符串描述。
例如：["步骤1: 搜索相关信息", "步骤2: 分析搜索结果", "步骤3: 整理成报告"]

只返回 JSON 数组，不要其他内容。"""
        
        messages = [
            SystemMessage("你是一个任务规划专家，善于将复杂任务分解为可执行的步骤。"),
            UserMessage(plan_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages, temperature=temperature)
            # print(response)
            # 解析 JSON
            plan = self.json_parser.parse(response)
            if isinstance(plan, list):
                return plan
        except Exception as e:
            logger.warning(f"解析计划失败: {e}，使用默认单步计划")
        
        return [f"完成任务: {query}"]
    
    def _execute_step(self, step: str, temperature: float) -> str:
        """执行单个步骤"""
        if self.enable_tool and self.tool_registry:
            # 使用工具执行
            messages: list = []
            return super().invoke_with_tool(step, messages, max_iter=5, temperature=temperature)
        else:
            # 直接使用 LLM
            messages = [
                SystemMessage(self.get_enhanced_prompt()),
                UserMessage(step)
            ]
            return self.llm.invoke(messages, temperature=temperature)
    
    def _should_replan(self, result: str) -> bool:
        """判断是否需要重新规划"""
        # 简单判断：如果结果包含失败、错误等关键词
        failure_keywords = ["失败", "错误", "无法", "不能", "找不到", "error", "failed"]
        return any(kw in result.lower() for kw in failure_keywords)
    
    def _replan(
        self, 
        original_query: str, 
        executed_results: List[Dict], 
        remaining_plan: List[str],
        temperature: float
    ) -> List[str]:
        """重新规划剩余步骤"""
        replan_prompt = f"""原始任务：{original_query}

已执行的步骤和结果：
{json.dumps(executed_results, ensure_ascii=False, indent=2)}

原计划的剩余步骤：
{json.dumps(remaining_plan, ensure_ascii=False, indent=2)}

根据已执行的结果，请重新规划剩余步骤。
以 JSON 数组格式返回新的步骤列表。"""
        
        messages = [
            SystemMessage("你是一个任务规划专家，需要根据执行情况调整计划。"),
            UserMessage(replan_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages, temperature=temperature)
            new_plan = self.json_parser.parse(response)
            if isinstance(new_plan, list):
                return new_plan
        except Exception as e:
            logger.warning(f"重新规划失败: {e}")
        
        return remaining_plan
    
    def _summarize_results(
        self, 
        query: str, 
        results: List[Dict], 
        temperature: float
    ) -> str:
        """汇总执行结果"""
        summary_prompt = f"""原始任务：{query}

执行记录：
{json.dumps(results, ensure_ascii=False, indent=2)}

请根据以上执行记录，给出最终的完整回答。"""
        
        messages = [
            SystemMessage("你是一个助手，需要根据任务执行记录给出最终回答。"),
            UserMessage(summary_prompt)
        ]
        
        return self.llm.invoke(messages, temperature=temperature)
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """获取执行日志"""
        return self.execution_log.copy()
    
    def clear_execution_log(self) -> None:
        """清空执行日志"""
        self.execution_log.clear()
        self.current_plan.clear()
