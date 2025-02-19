from pydantic import BaseModel, Field
from typing import List, Dict, Any
from typing import Optional

class KnowledgeTestRequest(BaseModel):
    type: Optional[str] = Field(default="aaa", description="test type")
    
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class SplitTextRequest(BaseModel):
    type: str
    top_k: int = 3

class ChatRequest(BaseModel):
    query: str