# Thin forwarder to keep public import stable during refactor.
# Consumers should import: from k2_quant.pages.analysis.widgets.chart import ChartWidget
from k2_quant.pages.analysis.widgets.chart_widget import ChartWidget as _LegacyChartWidget  # type: ignore


class ChartWidget(_LegacyChartWidget):
    """Public facade for the chart widget.

    This forwards to the existing implementation for now so the codebase can
    gradually migrate internals into the chart/ package without breaking imports.
    """

    pass
