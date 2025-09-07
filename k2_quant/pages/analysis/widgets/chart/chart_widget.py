# Thin forwarder to keep public import stable during refactor.
# Consumers should import: from k2_quant.pages.analysis.widgets.chart import ChartWidget
from .main import ChartWidget as _MainChartWidget  # type: ignore


class ChartWidget(_MainChartWidget):
    """Public facade for the chart widget (for backward compatibility)."""
    pass
