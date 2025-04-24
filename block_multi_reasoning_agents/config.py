from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(..., description="OpenAI API Key")
    OPENAI_BASE_URL: str = Field(..., description="OpenAI Base URL")
    LLM_MODEL_NAME: str = Field(..., description="LLM Model Name")
    
    INDENT_API_KEY: str = Field(..., description="Indent API Key")
    INDENT_BASE_URL: str = Field(..., description="Indent Base URL")
    INDENT_MODEL_NAME: str = Field(..., description="Indent Model Name")
    
    VL_API_KEY: str = Field(..., description="VL API Key")
    VL_BASE_URL: str = Field(..., description="VL Base URL")
    VISION_MODEL_NAME: str = Field(..., description="Vision Model Name")
    
    class Config:
        env_file = ".env"

settings = Settings()