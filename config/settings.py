from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str  = "sk-Pd9jWpeUtXEYwXvERtooH0QaCH8554xUAXpGa6awqjXHPGBV"
    OPENAI_API_BASE: str = "https://api.zetatechs.com/v1" # 默认值https://api.zetatechs.com
    
    class Config:
        env_file = "config.env"
        env_file_encoding = 'utf-8'  # Add encoding specification


settings = Settings()
