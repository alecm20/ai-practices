from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from loguru import logger
from app.services.knowledge import knowledge_service
from app.schemas.knowledge import KnowledgeTestRequest, SearchRequest, ChatRequest
from typing import Optional

router = APIRouter()

@router.post("/test")
async def test(
):
    """test"""
    try:
        return await knowledge_service.test()
    except Exception as e:
        logger.error(f"Failed to test: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create_embedding")
async def create_embedding(
    file: Optional[UploadFile] = File(None),
    type: str = Form("txt"),
    url: Optional[str] = Form(None)
):
    """上传 txt 文件，创建向量存储到 LanceDB"""
    return await knowledge_service.create_embedding(file, type, url)

@router.post("/search")
async def search_text(request: SearchRequest):
    return await knowledge_service.search(request.query, request.top_k)

@router.post("/split_text")
async def split_text(
    file: Optional[UploadFile] = File(None),
    type: str = Form("txt"),
    url: Optional[str] = Form(None)):
    print("file type url", file, type, url)
    return await knowledge_service.split_text(file, type, url)

@router.post("/chat_with_knowledge")
async def chat_with_knowledge(
    request: ChatRequest
):
    return await knowledge_service.chat_with_knowledge(request.query)
