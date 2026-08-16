"""House figure style, defined once so it cannot drift across scripts.

City identity belongs in facet titles or axis labels, not arbitrary marker
shapes. Estimates use circles and light interval marks. One saturated accent
is reserved for substantive reference marks and map data layers.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

GRAYS = {
    "mumbai": "0.0",
    "navi_mumbai": "0.35",
    "bangalore": "0.6",
    "delhi": "0.78",
}
ACCENT = "#800000"  # reference lines and map data layers only
BAR_GRAY = "0.45"
BAND_GRAY = "0.88"

# Sequential ramp for magnitude layers (heatmaps, share-coded points): the
# accent hue from near-white to full, replacing multi-hue colormaps.
ACCENT_CMAP = LinearSegmentedColormap.from_list("accent_seq", ["#f7f0f0", ACCENT])

CITY_LABELS = {
    "mumbai": "Mumbai",
    "navi_mumbai": "Navi Mumbai",
    "bangalore": "Bangalore",
    "delhi": "Delhi",
}

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style() -> None:
    plt.rcParams.update(RC)
