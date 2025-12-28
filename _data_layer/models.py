from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

@dataclass
class Artifact:
    id: str
    stage: str
    backend: str
    data_ref: str            # path or GridFS id
    parameters: Dict[str, Any]
    parents: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class Text_Element:
    def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.metadata = metadata or {}