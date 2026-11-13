"""Render archived score analyses; never reads answers or reruns judges.

Inputs are byte-for-byte snapshots of docs/figures/*.csv in
trec-auto-judge-26 commit 89f38b1ab94e9d43a0101095be0f35f64ca2e45f.
Run with Python providing NumPy, pandas, Matplotlib, and Linux Libertine fonts.
"""
from pathlib import Path
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "docs" / "figures"  # from scripts/agreement_heatmap.py, topic_alpha.py
FIGURES = HERE.parent.parent / "trec-autojudge-26-report" / "figures"  # sibling report repo
HASHES = {
    "topic_alpha.csv": "345089a31ba3fa3564cc49029e66dd153130ffa41c3ecb1e44e18f7d6e777f02",
    "judge_agreement_tau_b.csv": "6f2f50a0bab4081f78891ef6b2c2ee08ff18ca1617c4632356f88a57d358d80f",
    "judge_agreement_tauap_b.csv": "0f323110ac1d422289dfa5c25b163f1aab8f942a3f5da41a87f0b25ba0bc2c1a",
}
for name, digest in HASHES.items():
    assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest

for name in ["LinLibertine_R.ttf", "LinLibertine_RI.ttf", "LinLibertine_RB.ttf", "LinLibertine_RBI.ttf"]:
    path = Path.home() / "Library/Fonts" / name
    if path.exists():
        font_manager.fontManager.addfont(str(path))
font_manager.findfont("Linux Libertine", fallback_to_default=False)
plt.rcParams.update({
    "font.family": "Linux Libertine", "font.size": 8,
    "mathtext.fontset": "custom", "mathtext.rm": "Linux Libertine",
    "mathtext.it": "Linux Libertine:italic", "mathtext.bf": "Linux Libertine:bold",
    "mathtext.fallback": None, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.unicode_minus": True, "axes.linewidth": .5,
})
BLUE, PURPLE = "#1a80ba", "#d7b4f3"
PDF_METADATA = {"CreationDate": None, "ModDate": None}

def assert_text_inside(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in fig.findobj(Text):
        if label.get_visible() and label.get_text():
            bounds = label.get_window_extent(renderer)
            assert bounds.x0 >= 0 and bounds.y0 >= 0, label.get_text()
            assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height, label.get_text()

alpha = pd.read_csv(DATA / "topic_alpha.csv", index_col=0)
assert alpha.shape == (119, 2) and alpha.index.is_unique
assert list(alpha) == ["arena judges (3)", "all measures (12)"]
assert np.isfinite(alpha.to_numpy()).all()
# Independently verified ordinal-alpha summaries, archived in report-support/evidence/.
assert np.allclose(alpha.mean(), [.7633154350835766, .24392403627720197], atol=1e-12, rtol=0)
assert np.allclose(alpha.median(), [.7800201229767371, .24744149911273683], atol=1e-12, rtol=0)
assert np.allclose(alpha.min(), [.40359565540387055, .09111114408136345], atol=1e-12, rtol=0)
assert np.allclose(alpha.max(), [.9002037002100731, .3399312304103992], atol=1e-12, rtol=0)

fig, ax = plt.subplots(figsize=(4.2, 2.3))
fig.subplots_adjust(left=.12, right=.99, bottom=.20, top=.97)
positions = [1, 1.7]
boxes = ax.boxplot([alpha.iloc[:, 0], alpha.iloc[:, 1]], positions=positions,
           widths=.28, showmeans=True, whis=1.5, zorder=3,
           boxprops={"color": "black", "linewidth": 1},
           medianprops={"color": "C1", "linewidth": 1.4},
           meanprops={"marker": "^", "markersize": 6,
                      "markerfacecolor": "C2", "markeredgecolor": "C2"},
           flierprops={"marker": "o", "markersize": 4,
                       "markerfacecolor": "none", "markeredgecolor": "black"},
           whiskerprops={"color": "black", "linewidth": 1},
           capprops={"color": "black", "linewidth": 1})
topic_points = []
for i, (position, column) in enumerate(zip(positions, alpha)):
    jitter = np.random.default_rng(1).normal(position, .06, len(alpha))
    assert len(jitter) == 119 and ((jitter > .72) & (jitter < 1.98)).all()
    topic_points.append(ax.scatter(jitter, alpha[column], s=8, alpha=.4,
                                   color=f"C{i}", zorder=2))
expected_outliers = [{"rag2026-10", "rag2026-77"},
                     {"rag2026-36", "rag2026-84", "rag2026-113"}]
annotations = []
for i, (position, column) in enumerate(zip(positions, alpha)):
    flier_values = boxes["fliers"][i].get_ydata()
    outliers = alpha[column][alpha[column].isin(flier_values)].sort_values(ascending=False)
    assert set(outliers.index) == expected_outliers[i]
    assert np.array_equal(np.sort(outliers), np.sort(flier_values))
    ring_coordinates = []
    for j, (topic, value) in enumerate(outliers.items()):
        point = topic_points[i].get_offsets()[alpha.index.get_loc(topic)]
        assert point[1] == value
        ring_coordinates.append(point)
        offset = (-25, 0) if i == 0 else (-52 if j == 0 else -40, 9 - 9 * j)
        annotations.append(ax.annotate(
            "T" + topic.removeprefix("rag2026-"), tuple(point),
            xytext=offset, textcoords="offset points", ha="right",
            va="center" if i == 0 else "bottom", fontsize=8,
            arrowprops={"arrowstyle": "-", "color": ".55", "linewidth": .5},
            zorder=4))
        assert np.array_equal(annotations[-1].xy, point)
    ring_coordinates = np.array(ring_coordinates)
    boxes["fliers"][i].set_data(ring_coordinates[:, 0], ring_coordinates[:, 1])
    assert np.array_equal(np.column_stack(boxes["fliers"][i].get_data()), ring_coordinates)
    assert np.array_equal(np.sort(boxes["fliers"][i].get_ydata()), np.sort(flier_values))
assert len(annotations) == 5
ax.set_xlim(.72, 1.98)
ax.set_xticks(positions, ["Arena judges\n(Plain, V1, V2)", "All outputs\n(12 measures)"])
ax.set_ylim(0, 1)
ax.set_yticks([0, .25, .5, .75, 1], ["0", ".25", ".50", ".75", "1"])
ax.set_ylabel("Ordinal Krippendorff's α", fontsize=9)
ax.text(.98, .98, "119 topics", transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
ax.tick_params(axis="both", labelsize=8.5, length=0, pad=3)
ax.grid(True, color="gray", alpha=.3, linewidth=.6)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set(color="black", linewidth=.8)
assert_text_inside(fig)
renderer = fig.canvas.get_renderer()
label_bounds = [Text.get_window_extent(a, renderer) for a in annotations]
assert all(not a.overlaps(b) for i, a in enumerate(label_bounds) for b in label_bounds[i + 1:])
fig.savefig(FIGURES / "agreement-boxplot.pdf", metadata=PDF_METADATA)
plt.close(fig)

matrices = [pd.read_csv(DATA / name, index_col=0)
            for name in ["judge_agreement_tau_b.csv", "judge_agreement_tauap_b.csv"]]
columns = list(matrices[0])
for matrix in matrices:
    values = matrix.to_numpy()
    assert matrix.shape == (12, 12)
    assert list(matrix) == list(matrix.index) == columns
    assert np.isfinite(values).all() and np.abs(values).max() <= 1 + 1e-12
    assert np.allclose(values, values.T, atol=1e-12)
    assert np.allclose(np.diag(values), 1, atol=1e-12)

labels = ["Arena: Plain", "Arena: Rubric V1", "Arena: Rubric V2",
          "Plain: Citation precision", "Plain: Citation recall",
          "RAGree: Citation coverage", "RAGree: Citation precision", "RAGree: Claim F1",
          "RAGree: Faithfulness", "RAGree: Critical errors (inv.)",
          "RAGree: Estimated coverage", "RAGree: Evidence coverage"]
xlabels = labels.copy()
xlabels[9] = "RAGree: Critical errors"  # Inversion is stated in the shared row label and caption.
upper = np.triu(np.ones((12, 12), dtype=bool), 1)
lower = upper.T
values = np.full((12, 12), np.nan)
values[upper] = matrices[0].to_numpy()[upper]
values[lower] = matrices[1].to_numpy()[lower]
assert upper.sum() == lower.sum() == 66 and np.isnan(values).sum() == 12
assert np.isnan(np.diag(values)).all()
assert np.array_equal(values[upper], matrices[0].to_numpy()[upper])
assert np.array_equal(values[lower], matrices[1].to_numpy()[lower])
cmap = LinearSegmentedColormap.from_list("report_agreement", [PURPLE, "white", BLUE])
cmap.set_bad("white")
fig = plt.figure(figsize=(5.15, 4.5))
ax = fig.add_axes([1.50 / 5.15, 1.25 / 4.5, 3 / 5.15, 3 / 4.5])
im = ax.imshow(values, cmap=cmap, vmin=-1, vmax=1, interpolation="none")
ax.set_title(r"Upper: Kendall $\tau_b$; lower: top-weighted $\tau_{\mathrm{AP},b}$", fontsize=9, pad=4)
ax.set_xticks(range(12), xlabels, rotation=62, ha="right", rotation_mode="anchor")
ax.set_yticks(range(12), labels)
ax.tick_params(axis="both", which="major", labelsize=8, length=0, pad=2)
ax.set_xticks(np.arange(-.5, 12), minor=True)
ax.set_yticks(np.arange(-.5, 12), minor=True)
ax.grid(which="minor", color="white", linewidth=.45)
ax.tick_params(which="minor", length=0)
for boundary in [2.5, 4.5]:
    ax.axhline(boundary, color="#77838d", linewidth=.55)
    ax.axvline(boundary, color="#77838d", linewidth=.55)
annotations = []
for row, col in np.ndindex(values.shape):
    if row == col:
        continue
    rounded = round(float(values[row, col]), 2)
    label = "0" if rounded == 0 else f"{rounded:.2f}".replace("0.", ".").replace("-", "−")
    annotations.append(ax.text(col, row, label, ha="center", va="center", fontsize=8, color="#111111"))
assert len(annotations) == 132
assert all(label.get_position()[0] != label.get_position()[1] for label in annotations)
for spine in ax.spines.values():
    spine.set_visible(False)
cax = fig.add_axes([4.62 / 5.15, 1.25 / 4.5, .10 / 5.15, 3 / 4.5])
bar = fig.colorbar(im, cax=cax, ticks=[-1, -.5, 0, .5, 1])
bar.ax.tick_params(labelsize=8, length=2, width=.5, pad=2)
bar.outline.set_visible(False)
assert_text_inside(fig)
fig.savefig(FIGURES / "judge-agreement.pdf", metadata=PDF_METADATA)
plt.close(fig)
