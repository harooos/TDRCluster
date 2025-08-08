import os
from typing import List
from openai import OpenAI

class EmbeddingService:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
