from fastapi import APIRouter
from app.apis import knowledge

router = APIRouter(prefix="/api")
router.include_router(knowledge.router, prefix="/knowledge", tags=["knowledge"])
