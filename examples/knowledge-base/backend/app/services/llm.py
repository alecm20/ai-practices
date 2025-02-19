from openai import OpenAI
from config import get_settings
from loguru import logger
from typing import List, Dict, Any
from http import HTTPStatus
from pathlib import PurePosixPath
from urllib.parse import urlparse, unquote

settings = get_settings()

openai_client = None
if settings.openai_api_key:
   openai_client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url or "https://api.openai.com/v1")

class LLMService:
    def __init__(self):
        self.openai_client = openai_client
    
    async def generate_response(self, *, text_llm_model: str = None, messages: List[Dict[str, str]]) -> any:
        """生成 LLM 响应
        Args:
            messages: 消息列表
        Returns:
            Dict[str, Any]: 解析后的响应
        Raises:
            Exception: 请求失败或解析失败时抛出异常
        """

        text_client = self.openai_client
        
        if text_llm_model == None:
            text_llm_model = settings.text_llm_model
        response = text_client.chat.completions.create(
            model= text_llm_model,
            messages=messages,
        )
        try:
            content = response.choices[0].message.content
            return content
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            raise e

# 创建服务实例
llm_service = LLMService()
