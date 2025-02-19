from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.apis.router import router

app = FastAPI(
    title="knowledge-base Backend API",
    description="Backend API for knowledge-base application",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router)

@app.get("/")
async def root():
    return {
        "app_name": "knowledge-base Backend API",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
