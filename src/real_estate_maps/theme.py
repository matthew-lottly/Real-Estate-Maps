"""Look and feel of the maps (colours, fonts). Subclass or tweak to restyle everything."""
from __future__ import annotations

from dataclasses import dataclass, field

from matplotlib import font_manager


@dataclass
class MapTheme:
    paper: str = "#FBFAF7"
    ink: str = "#1B2430"
    muted: str = "#5C6773"
    faint: str = "#98A1AB"
    water: str = "#D5E3EC"
    land: str = "#F1EDE4"
    county_line: str = "#BDB5A7"
    zip_line: str = "#FFFFFF"
    nodata: str = "#E4E6E9"
    nodata_hatch: str = "#B3BAC3"
    accent: str = "#D1493B"
    highlight: str = "#1B2430"
    font_stack: tuple[str, ...] = ("Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")

    def font(self) -> str:
        """First installed family from the stack (maps still render with DejaVu Sans anywhere)."""
        installed = {f.name for f in font_manager.fontManager.ttflist}
        return next((f for f in self.font_stack if f in installed), "DejaVu Sans")
