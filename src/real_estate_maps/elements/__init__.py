"""Map elements. Each is a small class with one `draw(ctx)` method; compose them in `MapComposer`."""
from .layers import BaseLayers, ChoroplethLayer, CountyLabels, PlaceLabels, RankMarkers
from .furniture import Graticule, NorthArrow, ScaleBar
from .panels import HowToRead, InsetLocator, LegendBlock, RankingPanel, SourceNote, StatsPanel, TitleBlock
from .rankings import Rankings


class StandardElements:
    """The default set, in drawing order (the same order a cartographer would stack them)."""

    @staticmethod
    def poster():
        return [BaseLayers(), ChoroplethLayer(), CountyLabels(), Graticule(), PlaceLabels(), RankMarkers(), NorthArrow(), ScaleBar(),
                TitleBlock(), StatsPanel(), LegendBlock(), RankingPanel(), InsetLocator(), HowToRead(), SourceNote()]
