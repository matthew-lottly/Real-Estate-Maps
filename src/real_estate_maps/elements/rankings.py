from __future__ import annotations

from dataclasses import dataclass

from ..composer import RenderContext


@dataclass
class RankRow:
    rank: int
    zip: str
    value: float
    x: float
    y: float
    cls: int


class Rankings:
    """Highest / lowest ZIPs among those with enough sales. Shared by the map markers and the ranking panel."""

    @staticmethod
    def of(c: RenderContext, n: int = 5) -> tuple[list[RankRow], list[RankRow]]:
        g = c.zips[c.zips["eligible"]].copy()
        col = c.metric.column
        g = g.sort_values(col, ascending=False)

        def rows(frame):
            out = []
            for i, (_, r) in enumerate(frame.iterrows(), 1):
                p = r.geometry.representative_point()
                out.append(RankRow(i, r["zip"], float(r[col]), p.x, p.y, int(r["cls"])))
            return out

        return rows(g.head(n)), rows(g.tail(n).iloc[::-1])
