from pydantic import BaseModel
from typing import Dict, List


class PredictionRequest(BaseModel):
    records: List[Dict[str, float]]
