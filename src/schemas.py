from pydantic import BaseModel, Field
from typing import List, Optional

class HOI(BaseModel):
    human_id: int
    object_id: int
    verb: str
    score: float
    part: Optional[str] = None
    part_score: Optional[float] = None
    triplet: List[str] = Field(default_factory=list)

class FrameRecord(BaseModel):
    frame_index: int
    timestamp_ms: int
    humans: list
    objects: list
    hoi: List[HOI]
