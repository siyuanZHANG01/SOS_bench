import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 17,
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "axes.unicode_minus": False,
})

def nice_axis_max_log(vmax: float, base: float) -> float:
    if not np.isfinite(vmax) or vmax <= 0:
        return 1.0
    if base == 10:
        exp = math.ceil(math.log10(vmax))
        return float(10 ** exp)
    exp = math.ceil(math.log(vmax, base))
    return float(base ** exp)

def prepare_log_axis(vals: np.ndarray, base: float):
    vals = np.asarray(vals, dtype=float)
    pos = vals[np.isfinite(vals) & (vals > 0)]
    if pos.size == 0:
        vmax = 1.0
        vmin_eff = vmax / (base ** 3)
        vals_plot = np.full_like(vals, vmin_eff / base)
    else:
        vmax = float(pos.max())
        vmin_pos = float(pos.min())
        vmin_eff = min(vmin_pos, vmax / (base ** 3))
        eps = vmin_eff / base
        vals_plot = vals.copy()
        vals_plot[~np.isfinite(vals_plot) | (vals_plot <= 0)] = eps

    xmax = nice_axis_max_log(vmax, base)
    cut2 = xmax / base
    cut1 = xmax / (base ** 2)
    xmin = max(vmin_eff, xmax / (base ** 6))
    return vals_plot, float(xmin), float(xmax), float(cut1), float(cut2)

def generate_plot(points, mode, output_path):
    xs = np.array([p[1] for p in points], dtype=float)  # Global
    ys = np.array([p[2] for p in points], dtype=float)  # Local

    if mode in ("pgm", "sc"):
        x_base = 10.0
        y_base = 2.0
    else:  # mse, sd
        x_base = 10.0
        y_base = 10.0

    xs_plot, x_min, x_max, x_cut1, x_cut2 = prepare_log_axis(xs, x_base)
    ys_plot, y_min, y_max, y_cut1, y_cut2 = prepare_log_axis(ys, y_base)

    x_cut1, x_cut2 = 3e-4, 3e-3
    if mode == "sd":
        x_cut1, x_cut2 = 1e-3, 1e-2
        y_cut1, y_cut2 = 1e-3, 1e-2
        y_max = 1e-1
        y_pad_low = 0.8
        y_min = min(y_min if np.isfinite(y_min) and y_min > 0 else y_cut1,
                    y_cut1 * y_pad_low)
    if mode == "sc":
        y_cut1, y_cut2 = 2**7, 2**8
        y_max = 2**9
        y_pad_low = 0.8
        y_min = min(y_min if np.isfinite(y_min) and y_min > 0 else y_cut1,
                    y_cut1 * y_pad_low)
    if mode == "mse":
        y_cut1, y_cut2 = 1e-4, 1e-3
        posy_raw = ys[np.isfinite(ys) & (ys > 0)]
        if posy_raw.size:
            raw_ymax = float(posy_raw.max())
            tight_headroom_y = 1.06
            y_max = max(raw_ymax * tight_headroom_y, y_cut2 * 1.25, y_min * (1 + 1e-9))
        x_cut1, x_cut2 = 3e-4, 3e-3
        xs_plot = xs_plot.copy()
        xs_plot[~np.isfinite(xs_plot) | (xs_plot <= 0)] = x_min
        xs_plot[xs_plot < x_min] = x_min

    pad_low, pad_high = 0.8, 1.25
    x_min = min(x_min, x_cut1 * pad_low)
    x_max = max(x_max, x_cut2 * pad_high)

    tight_headroom = 1.06
    posx_raw = xs[np.isfinite(xs) & (xs > 0)]
    if posx_raw.size:
        raw_xmax = float(posx_raw.max())
        tight_xmax = raw_xmax * tight_headroom
        x_max = max(tight_xmax, x_cut2 * pad_high, x_min * (1 + 1e-9))

    Nx, Ny = 400, 400
    gx = np.geomspace(max(x_min, 1e-300), x_max, Nx) if x_min > 0 else np.geomspace(1e-12, x_max, Nx)
    gy = np.geomspace(max(y_min, 1e-300), y_max, Ny) if y_min > 0 else np.geomspace(1e-12, y_max, Ny)
    Xg, Yg = np.meshgrid(gx, gy)

    def _log01(v, vmin, vmax):
        lv  = np.log(v)
        lmn = np.log(max(vmin, 1e-300))
        lmx = np.log(max(vmax, 1e-299))
        return np.clip((lv - lmn) / (lmx - lmn), 0.0, 1.0)

    Zx = _log01(Xg, x_min, x_max)
    Zy = _log01(Yg, y_min, y_max)
    Z  = 0.5 * (Zx + Zy)
    cmap = plt.get_cmap("Reds")

    fig, ax = plt.subplots(figsize=(13.5, 10))
    ax.imshow(Z, extent=[x_min, x_max, y_min, y_max], origin="lower",
              aspect="auto", cmap=cmap, alpha=0.35)

    ax.set_xscale('log', base=x_base)
    ax.set_yscale('log', base=y_base)

    for xv in [x_cut1, x_cut2]:
        ax.axvline(x=xv, color="black", linewidth=0.8, linestyle="--", alpha=0.65)
    for yv in [y_cut1, y_cut2]:
        ax.axhline(y=yv, color="black", linewidth=0.8, linestyle="--", alpha=0.65)

    if mode == "pgm":
        ax.set_xlabel("Global = Std(start keys)")
        ax.set_ylabel("Local  = Avg Std(segments)")
        ax.set_title("PGM Hardness Map (L2)")
    elif mode == "sc":
        ax.set_xlabel("Global = Std(start keys)")
        ax.set_ylabel("Local  = Avg Std(segments)")
        ax.set_title("SC Hardness Map (L2)")
    elif mode == "mse":
        ax.set_xlabel("Global = Std(start keys)")
        ax.set_ylabel("Local  = Avg Std(segments)")
        ax.set_title("MSE Hardness Map (L2)")
    else:  # sd
        ax.set_xlabel("Global = Std(start keys)")
        ax.set_ylabel("Local  = Avg Std(segments)")
        ax.set_title("SD Hardness Map (L2)")

    def geo_mid(a, b, base):
        return math.sqrt(a * b) if a > 0 and b > 0 else (a + b) / 2.0

    x_pos_easy   = geo_mid(x_min,  x_cut1, x_base)
    x_pos_medium = geo_mid(x_cut1, x_cut2, x_base)
    x_pos_hard   = geo_mid(x_cut2, x_max,  x_base)
    y_pos_easy   = geo_mid(y_min,  y_cut1, y_base)
    y_pos_medium = geo_mid(y_cut1, y_cut2, y_base)
    y_pos_hard   = geo_mid(y_cut2, y_max,  y_base)

    ax.text(x_pos_easy,   y_min * (y_max / y_min) ** 0.05, "easy",   ha="center", va="bottom")
    ax.text(x_pos_medium, y_min * (y_max / y_min) ** 0.05, "medium", ha="center", va="bottom")
    ax.text(x_pos_hard,   y_min * (y_max / y_min) ** 0.05, "hard",   ha="center", va="bottom")

    ax.text(x_min * (x_max / x_min) ** 0.05, y_pos_easy,   "easy",   rotation=90, ha="left", va="center")
    ax.text(x_min * (x_max / x_min) ** 0.05, y_pos_medium, "medium", rotation=90, ha="left", va="center")
    ax.text(x_min * (x_max / x_min) ** 0.05, y_pos_hard,   "hard",   rotation=90, ha="left", va="center")

    COLOR_EASY   = "#4E79A7"
    COLOR_MEDIUM = "#EDC948"
    COLOR_HARD   = "#E15759"
    color_map = {"easy": COLOR_EASY, "medium": COLOR_MEDIUM, "hard": COLOR_HARD}
    marker_map = {"easy": "o", "medium": "^", "hard": "s"}

    def _level(val, c1, c2):
        if not np.isfinite(val):
            return "medium"
        if val < c1:
            return "easy"
        elif val > c2:
            return "hard"
        else:
            return "medium"

    levels_x = np.array([_level(v, x_cut1, x_cut2) for v in xs], dtype=object)
    levels_y = np.array([_level(v, y_cut1, y_cut2) for v in ys], dtype=object)

    for lx in ("easy", "medium", "hard"):
        for ly in ("easy", "medium", "hard"):
            mask = (levels_x == lx) & (levels_y == ly)
            if mask.any():
                ax.scatter(
                    xs_plot[mask], ys_plot[mask],
                    s=200, marker=marker_map[lx],
                    c=color_map[ly], zorder=5, edgecolors="none", alpha=0.95
                )

    color_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color_map["easy"],   label="Local: easy",   markersize=10),
        Line2D([0], [0], marker="o", linestyle="", color=color_map["medium"], label="Local: medium", markersize=10),
        Line2D([0], [0], marker="o", linestyle="", color=color_map["hard"],   label="Local: hard",   markersize=10),
    ]
    shape_handles = [
        Line2D([0], [0], marker=marker_map["easy"],   linestyle="", color="#444444", label="Global: easy",   markersize=10),
        Line2D([0], [0], marker=marker_map["medium"], linestyle="", color="#444444", label="Global: medium", markersize=10),
        Line2D([0], [0], marker=marker_map["hard"],   linestyle="", color="#444444", label="Global: hard",   markersize=10),
    ]

    leg1 = ax.legend(handles=color_handles, loc="upper left", bbox_to_anchor=(1.005, 1.00),
                     title="Local (color)", frameon=False, borderaxespad=0.3)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=shape_handles, loc="upper left", bbox_to_anchor=(1.005, 0.70),
                     title="Global (shape)", frameon=False, borderaxespad=0.3)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.25, linestyle=":", linewidth=0.8, which="both")

    fig.tight_layout(rect=[0, 0, 0.94, 1])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
