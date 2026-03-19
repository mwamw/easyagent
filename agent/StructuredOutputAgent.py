"""
结构化输出 Agent

强制 LLM 输出符合指定 Schema 的结构化数据。
"""
from typing import Optional, Type, TypeVar, Generic, TYPE_CHECKING
from typing_extensions import override
import logging

from .BasicAgent import BasicAgent
from core.llm import EasyLLM
from core.Message import Message, UserMessage, SystemMessage, AssistantMessage
from core.Config import Config
from Tool.ToolRegistry import ToolRegistry
from output.base import BaseOutputParser, OutputParseError
from output.pydantic_parser import PydanticOutputParser
from pydantic import BaseModel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from memory.V2.MemoryManage import MemoryManage
    from context.manager import ContextManager

T = TypeVar('T', bound=BaseModel)


class StructuredOutputAgent(BasicAgent, Generic[T]):
    """
    结构化输出 Agent
    
    强制 LLM 输出符合指定 Pydantic 模型的结构化数据。
    自动重试解析失败的情况。
    
    Attributes:
        output_parser: 输出解析器
        max_retries: 最大重试次数
        
    Example:
        >>> from pydantic import BaseModel, Field
        >>> 
        >>> class PersonInfo(BaseModel):
        ...     name: str = Field(description="姓名")
        ...     age: int = Field(description="年龄")
        ...     occupation: str = Field(description="职业")
        >>> 
        >>> agent = StructuredOutputAgent(
        ...     name="extractor",
        ...     llm=llm,
        ...     output_model=PersonInfo
        ... )
        >>> result = agent.invoke("张三，25岁，是一名软件工程师")
        >>> print(result.name, result.age)
        张三 25
    """
    
    def __init__(
        self,
        name: str,
        llm: EasyLLM,
        output_model: Type[T],
        system_prompt: Optional[str] = None,
        enable_tool: bool = False,
        tool_registry: Optional[ToolRegistry] = None,
        description: Optional[str] = None,
        config: Optional[Config] = None,
        max_retries: int = 3,
        memory_manage: Optional["MemoryManage"] = None,
        context_manager: Optional["ContextManager"] = None,
        history_via_context_manager: bool = False,
    ):
        """
        初始化结构化输出 Agent
        
        Args:
            name: Agent 名称
            llm: LLM 实例
            output_model: Pydantic 输出模型类
            system_prompt: 额外的系统提示词
            enable_tool: 是否启用工具
            tool_registry: 工具注册表
            description: Agent 描述
            config: 配置
            max_retries: 解析失败时的最大重试次数
        """
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            enable_tool=enable_tool,
            tool_registry=tool_registry,
            description=description,
            config=config,
            memory_manage=memory_manage,
            context_manager=context_manager,
            history_via_context_manager=history_via_context_manager,
        )
        
        self.output_model = output_model
        self.output_parser = PydanticOutputParser(output_model)
        self.max_retries = max_retries
        
        logger.info(f"StructuredOutputAgent '{name}' 初始化完成，输出模型: {output_model.__name__}")
    
    @override
    def invoke(
        self, 
        query: str, 
        max_iter: int = 10, 
        temperature: float = 0.7, 
        **kwargs
    ) -> T:
        """
        执行结构化输出提取
        
        Args:
            query: 用户输入
            max_iter: 最大迭代次数（用于工具调用）
            temperature: 温度参数
            
        Returns:
            解析后的 Pydantic 模型实例
            
        Raises:
            OutputParseError: 多次重试后仍然解析失败
        """
        # 构建带格式要求的提示词
        format_instructions = self.output_parser.get_format_instructions()
        
        enhanced_query = f"""{query}

{format_instructions}"""
        
        last_error: Optional[Exception] = None
        last_output: str = ""
        
        for attempt in range(self.max_retries):
            try:
                if attempt == 0:
                    # 首次尝试
                    messages = [
                        SystemMessage(self._build_system_prompt()),
                        UserMessage(enhanced_query)
                    ]
                else:
                    # 重试：告知 LLM 上次的错误
                    fix_prompt = f"""上次输出格式不正确，解析失败。

错误信息: {last_error}

你上次的输出:
{last_output}

请严格按照以下格式重新输出：
{format_instructions}

原始请求: {query}"""
                    messages = [
                        SystemMessage(self._build_system_prompt()),
                        UserMessage(fix_prompt)
                    ]
                
                # 调用 LLM
                response = self.llm.invoke(messages, temperature=temperature)
                last_output = response
                
                # 尝试解析
                result = self.output_parser.parse(response)
                
                # 记录到历史
                self.add_message(UserMessage(query))
                self.add_message(AssistantMessage(response))
                
                logger.info(f"结构化输出解析成功 (尝试 {attempt + 1}/{self.max_retries})")
                return result
                
            except OutputParseError as e:
                last_error = e
                logger.warning(f"解析失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                continue
        
        # 所有重试都失败
        raise OutputParseError(
            f"经过 {self.max_retries} 次尝试仍无法解析输出",
            last_output
        )
    
    def invoke_raw(
        self, 
        query: str, 
        temperature: float = 0.7, 
        **kwargs
    ) -> str:
        """
        执行原始调用（不解析输出）
        
        用于调试或需要原始输出的场景。
        
        Args:
            query: 用户输入
            temperature: 温度参数
            
        Returns:
            LLM 原始输出
        """
        format_instructions = self.output_parser.get_format_instructions()
        enhanced_query = f"{query}\n\n{format_instructions}"
        
        messages = [
            SystemMessage(self._build_system_prompt()),
            UserMessage(enhanced_query)
        ]
        
        return self.llm.invoke(messages, temperature=temperature)
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        base_prompt = self.system_prompt or "你是一个精确的信息提取助手。"
        
        return f"""{base_prompt}

你的任务是从用户输入中提取结构化信息，并以 JSON 格式输出。

重要规则：
1. 只输出 JSON，不要有任何其他文字说明
2. 确保 JSON 格式正确，可以被解析
3. 所有必填字段都必须提供
4. 如果某个信息无法从输入中提取，使用合理的默认值或 null"""
    
    def get_schema(self) -> dict:
        """获取输出模型的 JSON Schema"""
        return self.output_parser.get_schema()
    
    def set_output_model(self, model: Type[T]) -> None:
        """
        更换输出模型
        
        Args:
            model: 新的 Pydantic 模型类
        """
        self.output_model = model
        self.output_parser = PydanticOutputParser(model)
        logger.info(f"输出模型已更换为: {model.__name__}")
    
    def __repr__(self) -> str:
        return f"StructuredOutputAgent(name={self.name}, model={self.output_model.__name__})"
