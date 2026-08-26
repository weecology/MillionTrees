"""What a loose IoU matching threshold permits that a strict one does not, drawn to scale.

Supplementary figure for the AP reporting decision in CLAUDE.md section 4. MillionTrees
reports **AP40** as the headline (predictions matched to ground truth at IoU 0.4, the same
IoU behind recall, mask-aware precision and F1) and **AP60** as its strict complement.
This figure converts that pair of thresholds into the two errors a reader can picture --
how much of a crown a prediction may miss, and how far it may slide -- and then into
metres, so the AP40 -> AP60 drop has a physical meaning.

Three panels:

  A  Nested / concentric error. The prediction is correctly centred but too small and sits
     entirely inside the ground truth, so IoU is just the area ratio and the side ratio is
     sqrt(IoU).
  B  Pure translation. Prediction and ground truth are the same size, offset along one
     axis by d; IoU = (L-d)/(L+d), so d = L(1-IoU)/(1+IoU).
  C  The same two tolerances in metres, for a crown 100 px across, as ground sample
     distance varies. Every tolerance is a fixed number of pixels, so metres = pixels x
     GSD and each line is a ray from the origin. The shaded wedge is the slack the loose
     threshold accepts and the strict one rejects.

Panels A and B are scale-free -- they hold at any resolution and any crown size, which is
the point. Panel C is a crown-size sweep expressed in the units a detector sees: fixing
the crown at 100 px and varying GSD varies the physical tree, from a 1 m sapling at 1 cm
to an unrealistic 50 m crown at 50 cm. Read it as "if a crown occupies 100 px in my tiles,
here is what the threshold costs on the ground", not as a resolution effect: the
tolerances depend only on physical crown size, and GSD cancels out of IoU entirely.

Usage (defaults to the project's own AP40 / AP60 pair):

    python scripts/make_iou_tolerance_figure.py \\
        --out docs/public/iou_tolerance_figure.png --also-svg \\
        --table-out notes/iou_threshold_tolerance_table.md

Any other pair works, e.g. the conventional 0.5 for comparison:

    python scripts/make_iou_tolerance_figure.py --loose 0.4 --strict 0.5 \\
        --out docs/public/iou_tolerance_figure_0.4_vs_0.5.png
"""
import argparse
import math

import matplotlib
matplotlib.use("Agg")
# Real <text> elements in the SVG so labels stay editable in Illustrator/Inkscape.
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Colours identify the loose and strict thresholds everywhere in the figure; gold always
# means "accepted by the loose threshold, rejected by the strict one".
C_LOOSE, C_STRICT = "#C1553B", "#2C6E8E"
C_GT = "#3A3A3A"
C_GAIN = "#E8B24A"

GSDS_CM = [1, 3, 5, 10, 30, 50]
CROWN_PX = 100


def nested_side(t):
    """Side of the largest fully-nested prediction that still scores IoU >= t, in units of L."""
    return math.sqrt(t)


def nested_margin(t):
    """Ground truth left uncovered on each side by that nested prediction, in units of L."""
    return (1 - nested_side(t)) / 2


def shift_tol(t):
    """Largest one-axis offset of a same-size prediction that still scores IoU >= t, in units of L."""
    return (1 - t) / (1 + t)


def sym_diff(t, r=1.0):
    """Largest symmetric-difference area at IoU >= t, in units of GT area, for pred/GT area ratio r.

    S = (A + B) (1 - t) / (1 + t) holds for any two shapes at any offset -- an IoU
    threshold is exactly a cap on disagreement area.
    """
    return (1 + r) * (1 - t) / (1 + t)


def draw_square(ax, cx, cy, side, **kw):
    ax.add_patch(mpatches.Rectangle((cx - side / 2, cy - side / 2), side, side, **kw))


def panel_nested(ax, loose, strict):
    """Concentric shrink: how much of the crown a prediction may miss and still match."""
    draw_square(ax, 0, 0, 1.0, facecolor="#F0F0F0", edgecolor=C_GT, lw=1.8, zorder=1)

    # Gold ring between the two limits: the extra shrink the loose threshold permits.
    draw_square(ax, 0, 0, nested_side(strict), facecolor=C_GAIN, alpha=0.45,
                edgecolor="none", zorder=2)
    draw_square(ax, 0, 0, nested_side(loose), facecolor="#F0F0F0", edgecolor="none",
                zorder=2)
    for t, c in ((loose, C_LOOSE), (strict, C_STRICT)):
        draw_square(ax, 0, 0, nested_side(t), facecolor="none", edgecolor=c, lw=2.0,
                    ls="--", zorder=3)

    # Margin bracket on the left edge, one arrow per threshold, labelled outside the box.
    for t, c, y in ((strict, C_STRICT, 0.20), (loose, C_LOOSE, -0.20)):
        m = nested_margin(t)
        ax.annotate("", xy=(-0.5, y), xytext=(-0.5 + m, y),
                    arrowprops=dict(arrowstyle="<->", color=c, lw=1.4))
        ax.plot([-0.5, -0.56], [y, y], color=c, lw=0.8, zorder=4)
        ax.text(-0.59, y, f"{m:.3f}L\nmissed", color=c, fontsize=8.5,
                va="center", ha="right", linespacing=1.25)

    ax.text(0, 0.56, "ground truth  (side L)", ha="center", va="bottom",
            fontsize=9, color=C_GT)
    ax.text(-0.2, -1.0, "side ratio = $\\sqrt{\\mathrm{IoU}}$", ha="center", va="top",
            fontsize=9.5, color="#555555")
    ax.set_title("A  Prediction nested inside GT", fontsize=10.5, loc="left", pad=8)
    ax.set_xlim(-1.02, 0.62)
    ax.set_ylim(-1.18, 0.72)


def panel_shift(ax, loose, strict):
    """Pure translation: how far a correctly-sized prediction may slide and still match."""
    draw_square(ax, 0, 0, 1.0, facecolor="#F0F0F0", edgecolor=C_GT, lw=1.8, zorder=1)

    # Shade only the extra slop the loose threshold permits -- the same quantity the wedge
    # in panel C measures, so the gold means one thing across the whole figure.
    d_loose, d_strict = shift_tol(loose), shift_tol(strict)
    for x0 in (-0.5, 0.5):
        ax.add_patch(mpatches.Rectangle((x0 + d_strict, -0.5), d_loose - d_strict, 1.0,
                                        facecolor=C_GAIN, alpha=0.45, edgecolor="none",
                                        zorder=2))

    for t, c in ((loose, C_LOOSE), (strict, C_STRICT)):
        draw_square(ax, shift_tol(t), 0, 1.0, facecolor="none", edgecolor=c, lw=2.0,
                    ls="--", zorder=3)

    for t, c, y in ((strict, C_STRICT, -0.62), (loose, C_LOOSE, -0.80)):
        d = shift_tol(t)
        for x in (0.0, d):
            ax.plot([x, x], [-0.5, y], color=c, lw=0.7, ls=":", zorder=4)
        ax.annotate("", xy=(0, y), xytext=(d, y),
                    arrowprops=dict(arrowstyle="<->", color=c, lw=1.4))
        ax.text(d + 0.05, y, f"d = {d:.3f}L", color=c, fontsize=8.5,
                va="center", ha="left")

    ax.text(0, 0.56, "ground truth  (side L)", ha="center", va="bottom",
            fontsize=9, color=C_GT)
    ax.text(0.2, -1.0, "$d = L\\,(1-\\mathrm{IoU})/(1+\\mathrm{IoU})$", ha="center",
            va="top", fontsize=9.5, color="#555555")
    ax.set_title("B  Same-size prediction, offset", fontsize=10.5, loc="left", pad=8)
    ax.set_xlim(-0.72, 1.18)
    ax.set_ylim(-1.18, 0.72)


def panel_meters(ax, loose, strict):
    """The two tolerances in metres for a 100 px crown, as GSD converts pixels to ground."""
    gsd_cm = list(range(0, 56))
    crown = [g / 100.0 * CROWN_PX for g in gsd_cm]  # crown width in metres

    series = [
        ("max centroid offset", shift_tol, "-"),
        ("uncovered margin per side", nested_margin, ":"),
    ]
    for label, fn, ls in series:
        y_loose = [c * fn(loose) for c in crown]
        y_strict = [c * fn(strict) for c in crown]
        ax.fill_between(gsd_cm, y_strict, y_loose, color=C_GAIN, alpha=0.22, lw=0, zorder=1)
        ax.plot(gsd_cm, y_loose, ls=ls, color=C_LOOSE, lw=2.0, zorder=3,
                label=f"{label} @ IoU {loose:g}")
        ax.plot(gsd_cm, y_strict, ls=ls, color=C_STRICT, lw=2.0, zorder=3,
                label=f"{label} @ IoU {strict:g}")
        for g in GSDS_CM:
            L = g / 100.0 * CROWN_PX
            ax.plot([g, g], [L * fn(strict), L * fn(loose)], color=C_GAIN,
                    lw=3.0, solid_capstyle="butt", zorder=2)
            ax.plot([g], [L * fn(loose)], "o", ms=4, color=C_LOOSE, zorder=4)
            ax.plot([g], [L * fn(strict)], "o", ms=4, color=C_STRICT, zorder=4)

    for g in GSDS_CM:
        ax.axvline(g, color="#CCCCCC", lw=0.7, zorder=0)

    ax.set_xlabel("ground sample distance (cm / px)"
                  "        at 100 px per crown, 5 cm/px is a 5 m crown", labelpad=8)
    ax.set_ylabel("tolerance on the ground (m)")
    ax.set_xlim(0, 52)
    ax.set_ylim(0, max(crown) * shift_tol(loose) * 1.06)
    ax.set_xticks(GSDS_CM + [20, 40])
    ax.set_title(f"C  A crown 100 px across: what IoU {loose:g} vs {strict:g} is worth in metres",
                 fontsize=10.5, loc="left", pad=8)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # Every tolerance is a fixed pixel count, so the metre value is just pixels x GSD.
    ax.text(0.985, 0.05,
            f"in pixel space, always:  offset {shift_tol(loose) * CROWN_PX:.0f} px vs "
            f"{shift_tol(strict) * CROWN_PX:.0f} px\n"
            f"                        margin {nested_margin(loose) * CROWN_PX:.0f} px vs "
            f"{nested_margin(strict) * CROWN_PX:.0f} px",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="#555555", family="monospace")


def build_table(loose, strict):
    rows = []
    for g_cm in GSDS_CM:
        L = g_cm / 100.0 * CROWN_PX
        rows.append(dict(
            gsd_cm=g_cm,
            crown_m=L,
            area_m2=L * L,
            margin_loose=L * nested_margin(loose),
            margin_strict=L * nested_margin(strict),
            shift_loose=L * shift_tol(loose),
            shift_strict=L * shift_tol(strict),
            dS=L * L * (sym_diff(loose) - sym_diff(strict)),
        ))
    return rows


def format_table(rows, loose, strict):
    head = (f"| GSD | Crown width | Crown area | Margin/side {loose:g} | {strict:g} | gain | "
            f"Max offset {loose:g} | {strict:g} | gain | Extra disagreement area |")
    sep = "|" + "---|" * 10
    out = [head, sep]
    for r in rows:
        out.append(
            f"| {r['gsd_cm']:g} cm | {r['crown_m']:.2f} m | {r['area_m2']:.0f} m² | "
            f"{r['margin_loose']:.2f} m | {r['margin_strict']:.2f} m | "
            f"+{r['margin_loose'] - r['margin_strict']:.2f} m | "
            f"{r['shift_loose']:.2f} m | {r['shift_strict']:.2f} m | "
            f"+{r['shift_loose'] - r['shift_strict']:.2f} m | "
            f"+{r['dS']:.1f} m² |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loose", type=float, default=0.4,
                    help="Loose IoU threshold, the project headline (default: 0.4 = AP40).")
    ap.add_argument("--strict", type=float, default=0.6,
                    help="Strict IoU threshold, the complement (default: 0.6 = AP60).")
    ap.add_argument("--out", default="docs/public/iou_tolerance_figure.png")
    ap.add_argument("--also-svg", action="store_true",
                    help="Write the .svg alongside a .png --out (vector for the supplement).")
    ap.add_argument("--table-out", default=None,
                    help="Write the markdown table here instead of stdout.")
    ap.add_argument("--fig-width", type=float, default=10.0)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if not 0 < args.loose < args.strict < 1:
        raise SystemExit(f"need 0 < loose < strict < 1, got {args.loose} and {args.strict}")
    loose, strict = args.loose, args.strict

    fig = plt.figure(figsize=(args.fig_width, args.fig_width * 0.72))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.30, wspace=0.10,
                          left=0.085, right=0.965, top=0.88, bottom=0.13)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    for ax in (ax_a, ax_b):
        ax.set_aspect("equal")
        ax.axis("off")
    panel_nested(ax_a, loose, strict)
    panel_shift(ax_b, loose, strict)
    panel_meters(fig.add_subplot(gs[1, :]), loose, strict)

    handles = [mpatches.Patch(facecolor="#F0F0F0", edgecolor=C_GT, label="ground truth"),
               mpatches.Patch(facecolor="none", edgecolor=C_LOOSE,
                              label=f"limit at IoU {loose:g}"),
               mpatches.Patch(facecolor="none", edgecolor=C_STRICT,
                              label=f"limit at IoU {strict:g}"),
               mpatches.Patch(facecolor=C_GAIN, alpha=0.45, edgecolor="none",
                              label=f"accepted at {loose:g}, rejected at {strict:g}")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.995))

    fig.savefig(args.out, dpi=args.dpi)
    print("wrote", args.out)
    if args.also_svg and args.out.endswith(".png"):
        svg = args.out[:-4] + ".svg"
        fig.savefig(svg)
        print("wrote", svg)

    table = format_table(build_table(loose, strict), loose, strict)
    if args.table_out:
        with open(args.table_out, "w") as f:
            f.write(table + "\n")
        print("wrote", args.table_out)
    else:
        print(table)


if __name__ == "__main__":
    main()
