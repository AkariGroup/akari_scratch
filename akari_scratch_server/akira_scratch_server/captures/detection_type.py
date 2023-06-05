from typing import List, Union
from typing_extensions import TypedDict
from pydantic import BaseModel

class DetectionResult(BaseModel):
    id: int
    name: str
    x: int
    y: int
    width: int
    height: int


class GetDetectionResponse(BaseModel):
    result: bool = False
    data: List[DetectionResult] = []
