# Facade package for the chart widget.
# Re-export the ChartWidget from the new implementation location.
from .main import ChartWidget  # type: ignore

__all__ = ["ChartWidget"]
