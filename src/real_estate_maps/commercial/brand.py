"""Brand kit: the colours, fonts and logo a client supplies. Every map reads these instead of hard-coded styling."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from matplotlib import font_manager


@dataclass
class Brand:
    name: str = "Sample Brand"
    primary: str = "#14304A"
    secondary: str = "#1F8A8A"
    accent: str = "#C9A24B"
    alert: str = "#D1493B"
    ink: str = "#18222D"
    muted: str = "#5C6773"
    paper: str = "#FFFFFF"
    land: str = "#F2EFE8"
    water: str = "#CFE0EA"
    heading_fonts: tuple[str, ...] = ("Inter", "Helvetica Neue", "Arial", "DejaVu Sans")
    body_fonts: tuple[str, ...] = ("Inter", "Helvetica Neue", "Arial", "DejaVu Sans")
    logo: str | None = None            # path to a PNG, relative to the brand file
    sample: bool = True                # the built-in kit is a stand-in; real client kits set this False
    categories: dict = field(default_factory=lambda: {"retail": "#C9A24B", "restaurant": "#D1493B", "employer": "#1F8A8A", "park": "#5B9A5B", "transit": "#6C55A2", "health": "#6C55A2"})

    @staticmethod
    def _pick(stack) -> str:
        installed = {f.name for f in font_manager.fontManager.ttflist}
        return next((f for f in stack if f in installed), "DejaVu Sans")

    @property
    def heading(self) -> str:
        return self._pick(self.heading_fonts)

    @property
    def body(self) -> str:
        return self._pick(self.body_fonts)

    @classmethod
    def load(cls, path: Path) -> "Brand":
        d = json.loads(Path(path).read_text())
        for k in ("heading_fonts", "body_fonts"):
            if k in d:
                d[k] = tuple(d[k])
        if d.get("logo"):
            d["logo"] = str(Path(path).parent / d["logo"])
        d.setdefault("sample", False)
        return cls(**d)
