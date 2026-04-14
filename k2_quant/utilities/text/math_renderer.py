"""
LaTeX-to-QPixmap renderer using matplotlib's built-in mathtext engine.

No external dependencies beyond matplotlib (already required by the project).
Falls back gracefully: returns None if rendering fails so callers can
use the old MathFormatter text approximation.
"""

from io import BytesIO
from typing import Optional

from PyQt6.QtGui import QPixmap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def latex_to_pixmap(
    latex: str,
    fontsize: int = 14,
    dpi: int = 150,
    color: str = "#e0e0e0",
    max_width: int = 900,
) -> Optional[QPixmap]:
    """Render a LaTeX string to a ``QPixmap`` suitable for embedding in a QLabel.

    Parameters
    ----------
    latex : str
        Raw LaTeX (without ``$`` delimiters -- they are added internally).
    fontsize, dpi : int
        Control the resolution and size of the output.
    color : str
        Text colour (hex or named).
    max_width : int
        If the rendered image exceeds this width it is scaled down.

    Returns
    -------
    QPixmap or None
        The rendered equation, or ``None`` on failure.
    """
    try:
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.patch.set_alpha(0.0)
        text_obj = fig.text(
            0, 0, f"${latex}$",
            fontsize=fontsize,
            color=color,
            usetex=False,
        )

        buf = BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            transparent=True,
            pad_inches=0.08,
        )
        plt.close(fig)
        buf.seek(0)

        pm = QPixmap()
        pm.loadFromData(buf.read())
        buf.close()

        if pm.isNull():
            return None

        if pm.width() > max_width > 0:
            pm = pm.scaledToWidth(max_width)

        return pm
    except Exception:
        return None
