from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "knowledge-base"
    debug: bool = True
    version: str = "1.0.0"
    
    # provider configuration
    text_provider: str = "openai"
    image_provider: str = "openai"

    # base url configuration
    openai_base_url: str = "https://api.openai.com/v1"

    # api key
    openai_api_key: str = ""

    text_llm_model: str = "gpt-4o"
    image_llm_model: str = "dall-e-3"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
