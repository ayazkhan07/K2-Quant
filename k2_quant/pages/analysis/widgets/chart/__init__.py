# Facade package for the chart widget.
# During the migration, re-export the existing ChartWidget so imports can be updated incrementally.
from k2_quant.pages.analysis.widgets.chart_widget import ChartWidget  # type: ignore

__all__ = ["ChartWidget"]
