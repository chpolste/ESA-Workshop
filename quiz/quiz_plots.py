#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

x = np.linspace(-2, 2, 101)
y = np.linspace(-1, 1, 101)
xx, yy = np.meshgrid(x, y)


def esa_corr(source, target):
    src = source - source.mean(axis=0)
    tgt = target - target.mean()
    # Standard deviation
    std_src = np.sqrt( np.sum(src * src, axis=0) )
    std_tgt = np.sqrt( np.sum(tgt * tgt) )
    # Covariance, normalized by product of standard deviations = correlation
    cov = np.sum((src * tgt[:,None,None]), axis=0)
    return cov / std_src / std_tgt


def make_source(x=0., amplitude=10.):
    return amplitude * np.exp(-8*(xx - x)**2 - 8*yy**2) + 0.1*np.random.random(xx.shape)


# Colormap setup for correlation-based sensitivity maps (based on Magnusson 2017)
corr_kwargs = {
    "levels": [-1., -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.],
    "colors": ["#6D41A3", "#3758AF", "#57C9F5", "#FFFFFF", "#F9C719", "#F68026", "#EF282E"]
}

# Text annotations setup
text_kwargs = {
    "fontsize": 22,
    "fontweight": "bold",
    "ha": "center",
    "va": "center"
}

# Source field contourf setup
src_kwargs = {
    "cmap": "cubehelix_r",
    "levels": [0, 2, 4, 6, 8, 10, 12],
    "extend": "max"
}

def get_lomdhi(n):
    # Pick out first, middle and last ensemble member
    lo = 0
    md = n//2
    hi = n-1
    # Highlight picked members with color...
    cs = ["#999"] * n
    cs[lo] = "blue"
    cs[md] = "black"
    cs[hi] = "red"
    # ...and size
    ss = [10] * n
    ss[lo] = ss[md] = ss[hi] = 40
    return lo, md, hi, cs, ss


def ens_figure(src, tgt):
    fig = plt.figure(figsize=(12, 6))
    assert src.shape[0] == tgt.size
    n = src.shape[0]
    lo, md, hi, cs, ss = get_lomdhi(n)
    mem = np.arange(1, n+1)

    # Member number vs target value
    ax_tgt = fig.add_axes((0.05, 0.62, 0.4, 0.3))
    ax_tgt.scatter(mem, tgt, s=ss, c=cs)
    ax_tgt.set_xticks(mem[4::5])
    ax_tgt.set_title("Target Values", loc="left", fontsize="x-large")

    ax_lo = fig.add_axes((0.05, 0.25, 0.27, 0.27))
    ax_md = fig.add_axes((0.37, 0.25, 0.27, 0.27))
    ax_hi = fig.add_axes((0.68, 0.25, 0.27, 0.27))
    for i, ax, color in zip([lo, md, hi], [ax_lo, ax_md, ax_hi], ["blue", "black", "red"]):
        cf = ax.contourf(xx, yy, src[i], **src_kwargs)
        for spine in ["top", "bottom", "right", "left"]:
            ax.spines[spine].set_color(color)
            ax.spines[spine].set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])
    ax_lo.text(-1.5, -0.8, f"{tgt[lo]:.0f}", color="blue", **text_kwargs)
    ax_md.text(   0, -0.8, f"{tgt[md]:.0f}", color="black", **text_kwargs)
    ax_hi.text( 1.5, -0.8, f"{tgt[hi]:.0f}", color="red", **text_kwargs)
    
    cax = fig.add_axes((0.55, 0.62, 0.4, 0.05))
    plt.colorbar(cf, cax=cax, orientation="horizontal", spacing="proportional")
    fig.text(0.55, 0.72, "Source Field", fontsize="x-large")

    # Show part of all member source fields at the bottom
    for i in range(n):
        ax = fig.add_axes((0.05+0.87*i/(n-1), 0.05, 0.03, 0.1))
        ax.contourf(xx[:,35:-35], yy[:,35:-35], src[i,:,35:-35], **src_kwargs)
        if (i+1)%5 == 0:
            ax.set_title(str(i+1), loc="left")
        ax.set_xticks([])
        ax.set_yticks([])

    # Lines from small previews to picked out members
    fig.add_artist(plt.Line2D([0.065, 0.1], [0.15, 0.25], color="blue"))
    fig.add_artist(plt.Line2D([0.5,   0.5], [0.15, 0.25], color="black"))
    fig.add_artist(plt.Line2D([0.935, 0.9], [0.15, 0.25], color="red"))

    return fig


def quiz_figure(src, tgt, pois=None, show_esa=True):
    fig = plt.figure(figsize=(12, 6))
    assert src.shape[0] == tgt.size
    n = src.shape[0]
    lo, md, hi, cs, ss = get_lomdhi(n)

    ax_src = fig.add_axes((0.05, 0.55, 0.38, 0.38))
    ax_src.contour(xx, yy, src[lo], colors="blue", levels=[5, 11])
    ax_src.contour(xx, yy, src[md], colors="black", levels=[5, 11])
    ax_src.contour(xx, yy, src[hi], colors="red", levels=[5, 11])
    ax_src.text(-1.5, -0.8, f"{tgt[lo]:.0f}", color="blue", **text_kwargs)
    ax_src.text(   0, -0.8, f"{tgt[md]:.0f}", color="black", **text_kwargs)
    ax_src.text( 1.5, -0.8, f"{tgt[hi]:.0f}", color="red", **text_kwargs)
    ax_src.set_xticks([])
    ax_src.set_yticks([])

    if show_esa:
        ax_esa = fig.add_axes((0.05, 0.05, 0.38, 0.38))
        cf = ax_esa.contourf(xx, yy, esa_corr(src, tgt), **corr_kwargs)
        ax_esa.set_title("Sensitivity", loc="left", fontsize=18)
        ax_esa.set_xticks([])
        ax_esa.set_yticks([])

        pos = ax_esa.get_position()
        cax = fig.add_axes((pos.x1-0.2, pos.y1, 0.2, 0.03))
        cb = plt.colorbar(cf, cax=cax, orientation="horizontal", spacing="proportional")    
        cax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
        cb.set_ticks([-1, -0.5, 0, 0.5, 1])
        cax.set_xticklabels(["$-1$", "$-0.5$", "$0$", "$0.5$", "$1$"], fontsize="large")

    ax_abcd_pos = [
        (0.52, 0.55, 0.19, 0.38),
        (0.78, 0.55, 0.19, 0.38),
        (0.52, 0.05, 0.19, 0.38),
        (0.78, 0.05, 0.19, 0.38)
    ]
    if pois is not None:
        for poi, ax_pos, letter in zip(pois, ax_abcd_pos, "ABCD"):
            ax = fig.add_axes(ax_pos)
            # Get values for point of interest
            i, j = poi
            xs = src[:,j,i]
            ys = tgt
            # Show point cloud and regression line
            ax.scatter(xs, ys, s=ss, c=cs)
            reg = linregress(xs, ys)
            ax.plot(
                [min(xs), max(xs)],
                [reg.slope*min(xs)+reg.intercept, reg.slope*max(xs)+reg.intercept],
                linewidth=1.5,
                linestyle="dashed",
                color="#333"
            )
            ax.set_title(letter, loc="left", fontsize=18)
            ax.set_title(f"$r = {reg.rvalue:.2f}$", loc="right")
            ax_esa.text(x[i], y[j], letter, **text_kwargs)

    return fig



if __name__ == "__main__":

    # Ensemble size for case generation
    n = 50

    # Target values
    target = np.linspace(80, 120, n)
    
    # Source values and points of interest for the scatter plots
    cases = {
        "A-intensity-inc": {
            "source": np.asarray([make_source(0.0, 10+0.1*(i-n//2)) for i in range(n)]),
            "pois": [(50, 50), (66, 40), (90, 30)]
        },
        "B-intensity-dec": {
            "source": np.asarray([make_source(0.0, 10-0.1*(i-n//2)) for i in range(n)]),
            "pois": [(50, 50), (66, 40), (90, 30)]
        },
        "C-translation": {
            "source": np.asarray([make_source((i-n//2)/n, 10) for i in range(n)]),
            "pois": [(38, 50), (50, 50), (62, 50), (90, 30)]
        },
        "D-translation-inc": {
            "source": np.asarray([make_source((i-n//2)/n, 10+0.15*(i-n//2)) for i in range(n)]),
            "pois": [(38, 50), (50, 50), (62, 50), (90, 30)]
        },
        "E-translation-dec": {
            "source": np.asarray([make_source((i-n//2)/n, 10-0.15*(i-n//2)) for i in range(n)]),
            "pois": [(38, 50), (50, 50), (62, 50), (90, 30)]
        }
    }
    
    # ...

    for name, case in cases.items():
        fig = ens_figure(cases[name]["source"], target)
        fig.savefig(f"fig-{name}-ens.pdf")
        plt.close(fig)

        fig = quiz_figure(case["source"], target, show_esa=False)
        fig.savefig(f"fig-{name}-0.pdf")
        plt.close(fig)

        fig = quiz_figure(case["source"], target)
        fig.savefig(f"fig-{name}-1.pdf")
        plt.close(fig)

        for i in range(len(case["pois"])):
            fig = quiz_figure(case["source"], target, pois=case["pois"][:i+1])
            fig.savefig(f"fig-{name}-{i+2}.pdf")
            plt.close(fig)

