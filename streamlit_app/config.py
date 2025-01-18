from dotenv import find_dotenv
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OLLAMA_API_URL: str

    class Config:
        extra = "allow"


settings = Settings(_env_file=find_dotenv(), _env_file_encoding="utf-8")