from typing import List, Optional
from pydantic import BaseModel

class RetrievedDocument(BaseModel):
    fragmento: int
    contenido: str
    fuente: Optional[str] = "No especificada"
    pagina: Optional[int] = None


class RagResponse(BaseModel):
    answer: str
    documents: List[RetrievedDocument]
