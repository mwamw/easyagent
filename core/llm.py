
from .Message import Message
from typing import Optional
from openai import OpenAI
import logging
import os
logger=logging.getLogger(__name__)
class EasyLLM:
    def __init__(self,model:Optional[str]=None,temperature:Optional[float]=0.7,max_tokens:Optional[int]=None,api_key:Optional[str]=None,base_url:Optional[str]=None,timeout:Optional[int]=None,provide:Optional[str]="auto",**kwargs):
        self.provide=provide
        self.model=model or os.getenv("LLM_MODEL_ID")
        self.api_key=api_key or os.getenv("LLM_API_KEY")
        self.base_url=base_url or os.getenv("LLM_BASE_URL")
        self.temperature=temperature
        self.max_tokens=max_tokens
        self.timeout=timeout or int(os.getenv("LLM_TIMEOUT", 60))
        self.kwargs=kwargs
        if self.provide=="auto":
            self.provide=self.auto_detect_provider(api_key,base_url)
        self.resovle_api_key,self.resovle_base_url=self.resolve_api_key_and_base_url(self.api_key,self.base_url)
        
        if not model:
            self.model=self.get_default_model()
        if not self.resovle_api_key or not self.resovle_base_url:
            raise ValueError("API密钥和服务地址必须被提供或在.env文件中定义。")
        self.client=self.create_client()    

    def auto_detect_provider(self,api_key:Optional[str],base_url:Optional[str]):
        #根据环境变量来判断
        api_key=api_key or os.getenv("LLM_API_KEY")
        base_url=base_url or os.getenv("LLM_BASE_URL")
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("Google_API_KEY"):
            return "google"
        if os.getenv("DEEPSEEK_API_KEY"):
            return "deepseek"
        if os.getenv("DASHSCOPE_API_KEY"):
            return "qwen"
        if os.getenv("MODELSCOPE_API_KEY"):
            return "modelscope"
        if os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY"):
            return "kimi"
        if os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY"):
            return "zhipu"
        if os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_HOST"):
            return "ollama"
        if os.getenv("VLLM_API_KEY") or os.getenv("VLLM_HOST"):
            return "vllm"
        
        #根据base_url来判断

        if base_url:
            base_url_lower=base_url.lower()
            if "api.openai.com" in base_url_lower:
                return "openai"
            elif "google" in base_url_lower:
                return "google"
            elif "api.deepseek.com" in base_url_lower:
                return "deepseek"
            elif "dashscope.aliyuncs.com" in base_url_lower:
                return "qwen"
            elif "api-inference.modelscope.cn" in base_url_lower:
                return "modelscope"
            elif "api.moonshot.cn" in base_url_lower:
                return "kimi"
            elif "open.bigmodel.cn" in base_url_lower:
                return "zhipu"
            elif "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                # 本地部署检测 - 优先检查特定服务
                if ":11434" in base_url_lower or "ollama" in base_url_lower:
                    return "ollama"
                elif ":8000" in base_url_lower and "vllm" in base_url_lower:
                    return "vllm"

        #借助api_key来判断
        if api_key:
            api_key_lower=api_key.lower()
            if api_key_lower.startswith("sk-") and len(api_key_lower) > 50:
                # 可能是OpenAI、DeepSeek或Kimi，需要进一步判断
                pass
            elif api_key_lower.endswith(".") or "." in api_key_lower[-20:]:
                # 智谱AI的API密钥格式通常包含点号
                return "zhipu"   
        
        return "auto"

    def get_default_model(self):
        if self.provide=="openai":
            return "gpt-3.5-turbo"
        elif self.provide=="google":
            return "gemini-2.5-flash"
        elif self.provide=="deepseek":
            return "deepseek-chat"
        elif self.provide=="qwen":
            return "qwen2.5-coder:3b-instruct"
        elif self.provide=="modelscope":
            return "Qwen/Qwen2.5-VL-72B-Instruct"
        elif self.provide=="kimi":
            return "qwen2.5-coder:3b-instruct"
        elif self.provide=="zhipu":
            return "qwen2.5-coder:3b-instruct"
        elif self.provide=="ollama":
            return "qwen2.5-coder:3b-instruct"
        elif self.provide=="vllm":
            return "qwen2.5-coder:3b-instruct"
        else:
            return "qwen2.5-coder:3b-instruct"

    def resolve_api_key_and_base_url(self,api_key:Optional[str],base_url:Optional[str]):
        if api_key and base_url:
            return api_key,base_url
        if self.provide == "openai":
            resolve_api_key=self.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            resolve_base_url=self.base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            return resolve_api_key,resolve_base_url
        elif self.provide =="google":
            resolve_api_key=self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("LLM_API_KEY") 
            resolve_base_url=self.base_url or os.getenv("GOOGLE_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta"
            return resolve_api_key,resolve_base_url
        elif self.provide =="deepseek":
            resolve_api_key=self.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
            resolve_base_url=self.base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
            return resolve_api_key,resolve_base_url
        elif self.provide =="qwen":
            resolve_api_key=self.api_key or os.getenv("QWEN_API_KEY") or os.getenv("LLM_API_KEY")
            resolve_base_url=self.base_url or os.getenv("QWEN_BASE_URL") or "https://api.qwen.com/v1"
            return resolve_api_key,resolve_base_url
        elif self.provide =="modelscope":
            resolve_api_key=self.api_key or os.getenv("MODELSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            resolve_base_url=self.base_url or os.getenv("MODELSCOPE_BASE_URL") or "https://api-inference.modelscope.cn/v1/"
            return resolve_api_key,resolve_base_url
        elif self.provide =="kimi":
            resolve_api_key=self.api_key or os.getenv("KIMI_API_KEY") or os.getenv("LLM_API_KEY")
            resolve_base_url=self.base_url or os.getenv("KIMI_BASE_URL") or "https://api.moonshot.cn/v1"
            return resolve_api_key,resolve_base_url
        elif self.provide =="zhipu":
            resolve_api_key=self.api_key or os.getenv("ZHIPU_API_KEY") or os.getenv("LLM_API_KEY")
            resolve_base_url=self.base_url or os.getenv("ZHIPU_BASE_URL") or "https://open.bigmodel.cn/v1"
            return resolve_api_key,resolve_base_url
        elif self.provide =="ollama":
            resolve_api_key=self.api_key or os.getenv("OLLAMA_API_KEY") or os.getenv("LLM_API_KEY")
            resolve_base_url=self.base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
            return resolve_api_key,resolve_base_url
        elif self.provide =="vllm":
            resolve_api_key=self.api_key or os.getenv("VLLM_API_KEY") or os.getenv("LLM_API_KEY")
            resolve_base_url=self.base_url or os.getenv("VLLM_BASE_URL") or "http://localhost:8000"
            return resolve_api_key,resolve_base_url
        else:
            return api_key,base_url
    def create_client(self):
        return OpenAI(api_key=self.resovle_api_key, base_url=self.resovle_base_url, timeout=self.timeout)

    #流式输出
    def think(self,messages:list[dict[str,str]|Message],temperature:Optional[float]=None):
        """
        调用LLM的接口，返回流式输出
        这是主要的调用方法

        args：
            messages: 消息列表，每个消息是一个字典，包含role和content
            temperature: 温度参数，可选
        """
        temperature=temperature or self.temperature

        messages = [
            msg.to_dict() if isinstance(msg, Message) else msg 
            for msg in messages
        ]
        try:
            response=self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            logger.info("✅ 大语言模型响应成功:")
            for chunk in response:
                content=chunk.choices[0].delta.content or ""
                if content:
                    print(content,end="",flush=True)
                yield content

        except Exception as e:
            logger.error("❌ 大语言模型响应失败:")
            raise e

    def invoke(self,messages:list[dict[str,str]|Message],temperature:Optional[float]=None):
        """
        调用LLM的接口，返回非流式输出

        args：
            messages: 消息列表，每个消息是一个字典，包含role和content
            temperature: 温度参数，可选
        """
        temperature=temperature or self.temperature
        messages = [
            msg.to_dict() if isinstance(msg, Message) else msg 
            for msg in messages
        ]
        
        try:
            response=self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            logger.info("✅ 大语言模型响应成功:")
            return response.choices[0].message.content
        except Exception as e:
            logger.error("❌ 大语言模型响应失败:")
            raise e

    def stream_invoke(self,messages:list[dict[str,str]|Message],temperature:Optional[float]=None):
        """
        调用LLM的接口，返回流式输出

        args：
            messages: 消息列表，每个消息是一个字典，包含role和content
            temperature: 温度参数，可选
        """
        temperature=temperature or self.temperature

        yield from self.think(messages,temperature)

    def get_client(self):
        return self.client        

    def invoke_with_tools(self,messages:list[dict[str,str]|Message],tools:list[dict[str,str]],temperature:Optional[float]=None):
        """
        调用LLM的接口，返回非流式输出

        args：
            messages: 消息列表，每个消息是一个字典，包含role和content
            tools: 工具列表，每个工具是一个字典，包含name和description
            temperature: 温度参数，可选
        """
        temperature=temperature or self.temperature
        messages = [
            msg.to_dict() if isinstance(msg, Message) else msg 
            for msg in messages
        ]
                   
        try:
            response=self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            logger.info("✅ 大语言模型响应成功:")
            return response.choices[0].message
        except Exception as e:
            logger.error(f"调用LLM的messages:{messages[-1]}")

            logger.error("❌ 大语言模型响应失败:")
            raise e
    
