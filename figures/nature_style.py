"""Nature Methods figure styling defaults."""
import matplotlib.pyplot as plt
import matplotlib as mpl

def apply_nature_style():
    mm = 1 / 25.4
    mpl.rcParams.update({
        'font.size': 7,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'pdf.fonttype': 42,  # Ensure editable text in PDF/AI
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'lines.linewidth': 0.75,
        'savefig.transparent': True,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05
    })
    return mm