
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class Preset:
    version: int
    selected_variant: str
    variant_options: Dict[str, Any]
    prefer_1920: bool
    animated_fps: int
    animated_crf: int
    concurrency: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "Preset":
        obj = json.loads(s)
        return Preset(
            version=int(obj.get("version", 1)),
            selected_variant=str(obj["selected_variant"]),
            variant_options=dict(obj.get("variant_options", {})),
            prefer_1920=bool(obj.get("prefer_1920", True)),
            animated_fps=int(obj.get("animated_fps", 24)),
            animated_crf=int(obj.get("animated_crf", 20)),
            concurrency=int(obj.get("concurrency", 4)),
        )
