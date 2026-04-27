import matplotlib.pyplot as plt
import matplotlib as mpl
def apply_nature_style():
    mm = 1 / 25.4
    mpl.rcParams.update({
        'font.size': 7,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'pdf.fonttype': 42,  # 保证进 AI 可编辑
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'lines.linewidth': 0.75,
        # 这一条很重要：保证存图时边框不被裁掉，同时保留透明底
        'savefig.transparent': True,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05
    })
    return mm # 把毫米单位传回去