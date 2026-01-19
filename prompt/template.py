"""
提示词模板模块
"""
from typing import List, Dict, Any, Optional, Union
from string import Formatter
import re


class PromptTemplate:
    """
    提示词模板
    
    使用 Python 字符串格式化语法创建可复用的提示词模板。
    
    Example:
        >>> template = PromptTemplate(
        ...     template="你好，{name}！请帮我{task}。",
        ...     input_variables=["name", "task"]
        ... )
        >>> prompt = template.format(name="张三", task="写一封邮件")
        >>> print(prompt)
        你好，张三！请帮我写一封邮件。
    """
    
    def __init__(
        self,
        template: str,
        input_variables: Optional[List[str]] = None,
        partial_variables: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化提示词模板
        
        Args:
            template: 模板字符串
            input_variables: 输入变量列表（可选，自动检测）
            partial_variables: 预填充的变量
        """
        self.template = template
        self.partial_variables = partial_variables or {}
        
        # 自动检测输入变量
        if input_variables is None:
            self.input_variables = self._extract_variables()
        else:
            self.input_variables = input_variables
    
    def _extract_variables(self) -> List[str]:
        """从模板中提取变量名"""
        formatter = Formatter()
        variables = []
        for _, field_name, _, _ in formatter.parse(self.template):
            if field_name is not None and field_name not in self.partial_variables:
                # 处理 {var.attr} 形式，只取变量名
                var_name = field_name.split('.')[0].split('[')[0]
                if var_name and var_name not in variables:
                    variables.append(var_name)
        return variables
    
    def format(self, **kwargs) -> str:
        """
        格式化模板
        
        Args:
            **kwargs: 变量值
            
        Returns:
            格式化后的字符串
        """
        # 合并预填充变量和传入变量
        all_vars = {**self.partial_variables, **kwargs}
        
        # 检查必需变量
        missing = set(self.input_variables) - set(all_vars.keys())
        if missing:
            raise ValueError(f"缺少必需变量: {missing}")
        
        return self.template.format(**all_vars)
    
    def partial(self, **kwargs) -> "PromptTemplate":
        """
        创建部分填充的模板
        
        Args:
            **kwargs: 预填充的变量
            
        Returns:
            新的模板实例
        """
        new_partial = {**self.partial_variables, **kwargs}
        new_input_vars = [v for v in self.input_variables if v not in kwargs]
        return PromptTemplate(
            template=self.template,
            input_variables=new_input_vars,
            partial_variables=new_partial
        )
    
    def __add__(self, other: Union[str, "PromptTemplate"]) -> "PromptTemplate":
        """拼接模板"""
        if isinstance(other, str):
            return PromptTemplate(
                template=self.template + other,
                input_variables=self.input_variables,
                partial_variables=self.partial_variables
            )
        elif isinstance(other, PromptTemplate):
            combined_vars = list(set(self.input_variables + other.input_variables))
            combined_partial = {**self.partial_variables, **other.partial_variables}
            return PromptTemplate(
                template=self.template + other.template,
                input_variables=combined_vars,
                partial_variables=combined_partial
            )
        raise TypeError(f"无法与 {type(other)} 类型拼接")
    
    def __repr__(self) -> str:
        return f"PromptTemplate(variables={self.input_variables})"


class ChatPromptTemplate:
    """
    对话提示词模板
    
    用于构建多轮对话格式的提示词。
    
    Example:
        >>> template = ChatPromptTemplate.from_messages([
        ...     ("system", "你是一个{role}。"),
        ...     ("user", "{question}")
        ... ])
        >>> messages = template.format_messages(role="助手", question="你好")
    """
    
    def __init__(self, messages: List[Dict[str, Any]]):
        """
        初始化对话模板
        
        Args:
            messages: 消息模板列表，每个消息包含 role 和 content
        """
        self.messages = messages
        self.input_variables = self._extract_all_variables()
    
    def _extract_all_variables(self) -> List[str]:
        """提取所有消息中的变量"""
        all_vars = []
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                template = PromptTemplate(template=content)
                for var in template.input_variables:
                    if var not in all_vars:
                        all_vars.append(var)
        return all_vars
    
    @classmethod
    def from_messages(
        cls,
        messages: List[tuple]
    ) -> "ChatPromptTemplate":
        """
        从消息元组列表创建模板
        
        Args:
            messages: [(role, content), ...] 格式的消息列表
            
        Returns:
            ChatPromptTemplate 实例
        """
        msg_list = []
        for role, content in messages:
            msg_list.append({"role": role, "content": content})
        return cls(msg_list)
    
    def format_messages(self, **kwargs) -> List[Dict[str, str]]:
        """
        格式化消息列表
        
        Args:
            **kwargs: 变量值
            
        Returns:
            格式化后的消息列表
        """
        formatted = []
        for msg in self.messages:
            role = msg["role"]
            content = msg.get("content", "")
            
            if isinstance(content, str):
                template = PromptTemplate(template=content)
                formatted_content = template.format(**kwargs)
            else:
                formatted_content = content
            
            formatted.append({"role": role, "content": formatted_content})
        
        return formatted
    
    def partial(self, **kwargs) -> "ChatPromptTemplate":
        """创建部分填充的模板"""
        new_messages = []
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                template = PromptTemplate(template=content)
                partial_template = template.partial(**kwargs)
                new_messages.append({
                    "role": msg["role"],
                    "content": partial_template.template
                })
            else:
                new_messages.append(msg)
        return ChatPromptTemplate(new_messages)
    
    def __repr__(self) -> str:
        return f"ChatPromptTemplate(messages={len(self.messages)}, variables={self.input_variables})"
