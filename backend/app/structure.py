from typing import Dict, List, Optional
from pydantic import BaseModel, conlist

class InferResponseModel(BaseModel):
    top10_pred: list=[]
    confidence: list=[]
    status: str=''
