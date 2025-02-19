import os
import pyarrow as pa
import lancedb
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from config import get_settings
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLSectionSplitter, HTMLHeaderTextSplitter
import json
import requests
from app.services.llm import llm_service
from bs4 import BeautifulSoup
settings = get_settings()

# 配置参数
DB_DIR = "./lancedb_data"
TABLE_NAME = "vector_store"
VECTOR_DIMENSION = 1024  # 根据模型输出维度设置

class KnowledgeService:
    def __init__(self):
        """初始化数据库连接和模型"""
        self.db = lancedb.connect(DB_DIR)
        self._initialize_table()
        self.model = SentenceTransformer("BAAI/bge-m3")
        self.index_created = False  # 索引状态标记

    def _initialize_table(self):
        # 定义 Schema（PyArrow ≥ 12.0）
        # 指定vector的长度非常非常重要！
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("item", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=VECTOR_DIMENSION))  # 直接指定长度，一定要指定长度，否则搜索的时候会报错
        ])
        
        # 创建表
        if TABLE_NAME not in self.db.table_names():
            self.table = self.db.create_table(TABLE_NAME, schema=schema)
        else:
            self.table = self.db.open_table(TABLE_NAME)
    async def split_text(self, file: UploadFile = File(...), type: str = "txt", url: str = ""):
        if not file and not url:
            raise HTTPException(status_code=400, detail="file 和 url 至少提供一个")
        if type == "txt":
            return await self.get_texts_from_txt(file)
        elif type == "html":
            return await self.get_texts_from_html(url)
        else:
            raise HTTPException(status_code=400, detail="Invalid type")

    async def get_texts_from_txt(self, file: UploadFile = File(...)):
        """从 txt 文件中读取文本"""
        content = await file.read()
        content_str = content.decode("utf-8")
        
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=300,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.split_text(content_str)
        return texts
    async def get_texts_from_html(self, url: str):
        """从 html 文件中读取文本"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": url,  # 伪造来源
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        response = requests.get(url, headers=headers)

        response.raise_for_status()
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        # souhu的新闻网页格式
        post_body = soup.find(class_="post_body")
        texts = []
        # 提取其中的文本内容
        if post_body:
            text = post_body.get_text(separator="\n", strip=True)  # 使用换行符分隔段落
            text_splitter = RecursiveCharacterTextSplitter(
                # Set a really small chunk size, just to show.
                chunk_size=300,
                chunk_overlap=20,
                length_function=len,
                is_separator_regex=False,
            )
            texts = text_splitter.split_text(text)
        return texts

    async def create_embedding(self, file: UploadFile = File(...), type: str = "txt", url: str = ""):
        """处理文件并存储向量"""
        
        # 读取并处理文本
        texts = await self.split_text(file, type, url)
        
        # 生成向量
        embeddings = self.model.encode(texts)
        start_id = self.table.count_rows()
        # 构建数据
        data = [
            {
                "id": start_id + i,
                "item": line,
                "vector": emb
            }
            for i, (line, emb) in enumerate(zip(texts, embeddings))
        ]

        # 插入数据
        self.table.add(data)
        return {"message": f"成功存储{len(texts)}条数据"}

    async def search(self, query: str, top_k: int = 3):
        """执行向量搜索"""
        # 生成查询向量
        query_vector = self.model.encode(query)
        
        # 调整维度格式
        if query_vector.shape != (VECTOR_DIMENSION,):
            raise HTTPException(500, "查询向量维度错误")

        # 执行搜索
        try:
            results = self.table.search(query_vector) \
                .limit(top_k) \
                .to_pandas()
        except Exception as e:
            raise HTTPException(500, f"搜索失败: {str(e)}")

        # 格式化结果
        return [
            {"content": row["item"], "distance": row["_distance"]}
            for _, row in results.iterrows()
        ]
    
    async def chat_with_knowledge(self, query: str):
        """聊天"""
        # 生成查询向量
        results = await self.search(query, 3)
        context_str = "\n".join(item["content"] for item in results)
        messages = [
            {"role": "system", "content": "你是一个有知识的机器人"},
            {"role": "user", "content": context_str + "\n 基于上下文，回答这个问题： " + query}
        ]
        print("messages", messages)
        results = await llm_service.generate_response(text_llm_model="gpt-4o", messages=messages)
        return {
            "content": results
        }

# 单例服务实例
knowledge_service = KnowledgeService()