from pydantic import BaseModel


class ExtractResult(BaseModel):
    texts: list[str]

class ExtractResponse(BaseModel):
    results: list[str]
    