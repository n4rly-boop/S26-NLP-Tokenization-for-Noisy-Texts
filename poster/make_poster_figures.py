"""
Generate poster figures from results/tables/*.csv.

Outputs PDFs to poster/figures/:
  fertility_3panel.pdf     clean vs noisy vs +preprocess, per noise
  overlap_3panel.pdf       token-set Jaccard noisy vs preprocess, per noise
  f1_heatmap.pdf           12x4 mitigation matrix
  ablation_heatmap.pdf     4x4 component F1 contributions
  mechanism_scatter.pdf    LOO fertility_delta vs F1_delta with Pearson r
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES = os.path.join(ROOT, "results", "tables")
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

MODEL_ORDER = ["bert-base-uncased", "bert-base-cased", "gpt2", "google/byt5-small"]
MODEL_LABEL = {
    "bert-base-uncased": "bert-uncased",
    "bert-base-cased":   "bert-cased",
    "gpt2":              "gpt2",
    "google/byt5-small": "byt5-small",
}
NOISES = ["ocr", "asr", "social"]

INK   = "#222222"
MUTED = "#80A4A0"
COOL  = "#5B8BA0"
WARM  = "#D9A441"
BAD   = "#B5584D"
BG    = "#FAFAFA"

mpl.rcParams.update({
    "font.family":      "serif",
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "axes.edgecolor":   INK,
    "axes.labelcolor":  INK,
    "text.color":       INK,
    "xtick.color":      INK,
    "ytick.color":      INK,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "pdf.fonttype":     42,
    "savefig.bbox":     "tight",
    "savefig.facecolor": "white",
})

DIVERGE = LinearSegmentedColormap.from_list(
    "rwg", ["#B5584D", "#E8D4CC", "#FFFFFF", "#CFE2D4", "#5E8D6A"]
)
SEQ = LinearSegmentedColormap.from_list(
    "pale_blue_green", ["#F4F1E6", "#D6E4E0", "#A8C8C5", "#7BA7A2", "#4F8079"]
)


def load():
    story   = pd.read_csv(os.path.join(TABLES, "tokenizer_story.csv"))
    comb    = pd.read_csv(os.path.join(TABLES, "combined_results.csv"))
    f1c     = pd.read_csv(os.path.join(TABLES, "preprocess_contributions.csv"))
    tokc    = pd.read_csv(os.path.join(TABLES, "tokenizer_contributions.csv"))
    return story, comb, f1c, tokc


def fig_fertility(story, path):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    x = np.arange(len(MODEL_ORDER)); w = 0.27
    for ax, noise in zip(axes, NOISES):
        sub = story[story.noise == noise].set_index("model").reindex(MODEL_ORDER)
        ax.bar(x - w, sub.fertility_clean,      width=w, color=MUTED, label="clean",     edgecolor=INK, linewidth=0.5)
        ax.bar(x,     sub.fertility_noisy,      width=w, color=BAD,   label="noisy",     edgecolor=INK, linewidth=0.5)
        ax.bar(x + w, sub.fertility_preprocess, width=w, color=COOL,  label="+preprocess", edgecolor=INK, linewidth=0.5)
        ax.set_title(noise.upper(), color=INK, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("fertility (tokens / word)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncols=3, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(path, format="pdf")
    plt.close(fig)


def fig_overlap(story, path):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    x = np.arange(len(MODEL_ORDER)); w = 0.35
    for ax, noise in zip(axes, NOISES):
        sub = story[story.noise == noise].set_index("model").reindex(MODEL_ORDER)
        ax.bar(x - w/2, sub.overlap_noisy,      width=w, color=BAD,  label="noisy",       edgecolor=INK, linewidth=0.5)
        ax.bar(x + w/2, sub.overlap_preprocess, width=w, color=COOL, label="+preprocess", edgecolor=INK, linewidth=0.5)
        ax.set_title(noise.upper(), pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("token-set Jaccard vs clean")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncols=2, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(path, format="pdf")
    plt.close(fig)


def fig_f1_heatmap(comb, path):
    comb = comb.copy()
    comb["label"] = comb["model"].map(MODEL_LABEL) + " / " + comb["noise"].str.upper()
    order = []
    for m in MODEL_ORDER:
        for n in NOISES:
            order.append(f"{MODEL_LABEL[m]} / {n.upper()}")
    comb = comb.set_index("label").reindex(order)
    mat = comb[["f1_clean", "f1_noisy", "f1_preprocess", "f1_noisy_ft"]].to_numpy(dtype=float)
    col_labels = ["clean", "noisy", "+preprocess", "+noisy-ft"]

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(mat, cmap=SEQ, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.tick_params(length=0)

    best_col = np.nanargmax(mat[:, 1:], axis=1) + 1
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            color = "white" if v > 0.55 else INK
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=11)
        ax.add_patch(plt.Rectangle(
            (best_col[i] - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor=WARM, linewidth=2.0
        ))

    cbar = fig.colorbar(im, ax=ax, shrink=0.65, pad=0.02)
    cbar.ax.tick_params(length=0, labelsize=10)
    cbar.outline.set_visible(False)
    cbar.set_label("NER F1", fontsize=11)

    ax.set_title("Mitigation matrix: F1 per (model, noise) x intervention\n(outline = best non-clean intervention)",
                 fontsize=12, pad=10)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def fig_ablation_heatmap(f1c, path):
    pick = [
        ("asr",    "truecase",     "ASR\ntruecase"),
        ("ocr",    "charfix",      "OCR\ncharfix"),
        ("social", "spellcorrect", "Social\nspellcorrect"),
        ("ocr",    "spellcorrect", "OCR\nspellcorrect"),
    ]
    mat = np.full((len(MODEL_ORDER), len(pick)), np.nan)
    for i, m in enumerate(MODEL_ORDER):
        for j, (noise, comp, _) in enumerate(pick):
            row = f1c[(f1c.model == m) & (f1c.noise == noise) & (f1c.component == comp)]
            if len(row):
                mat[i, j] = float(row["contribution"].iloc[0])

    vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(mat, cmap=DIVERGE, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(pick)))
    ax.set_xticklabels([lbl for _, _, lbl in pick])
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_LABEL[m] for m in MODEL_ORDER])
    ax.tick_params(length=0)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color=INK if abs(v) < 0.25 else "white", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.ax.tick_params(length=0, labelsize=10)
    cbar.outline.set_visible(False)
    cbar.set_label(r"$\Delta$ F1 (component contribution)", fontsize=11)

    ax.set_title("Which preprocessing component matters?\n(F1(all) - F1(- component))",
                 fontsize=12, pad=10)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def fig_mechanism(f1c, tokc, path):
    merged = f1c.merge(
        tokc, on=["model", "noise", "component"], suffixes=("_f1", "_tok")
    )
    merged = merged.rename(columns={"contribution": "f1_contrib"})
    x = merged["overlap_contribution"].to_numpy(dtype=float)
    y = merged["f1_contrib"].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    sub = merged.loc[mask].reset_index(drop=True)

    if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
        r = float(np.corrcoef(x, y)[0, 1])
    else:
        r = float("nan")

    color_map = {"ocr": COOL, "asr": WARM, "social": MUTED}
    marker_map = {
        "bert-base-uncased": "o",
        "bert-base-cased":   "s",
        "gpt2":              "^",
        "google/byt5-small": "D",
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min() - 0.02, x.max() + 0.02, 50)
        ax.plot(xs, slope * xs + intercept, color=INK, linewidth=1.2, alpha=0.7, linestyle="--")

    for noise in NOISES:
        for m in MODEL_ORDER:
            sel = (sub["noise"] == noise) & (sub["model"] == m)
            if sel.any():
                ax.scatter(x[sel], y[sel],
                           color=color_map[noise], marker=marker_map[m],
                           s=90, edgecolor=INK, linewidth=0.6, alpha=0.9)

    from matplotlib.lines import Line2D
    noise_legend = [
        Line2D([], [], marker="o", linestyle="", color=color_map[n], markeredgecolor=INK,
               markersize=8, label=n.upper())
        for n in NOISES
    ]
    model_legend = [
        Line2D([], [], marker=marker_map[m], linestyle="", color="#BBBBBB", markeredgecolor=INK,
               markersize=8, label=MODEL_LABEL[m])
        for m in MODEL_ORDER
    ]
    leg1 = ax.legend(handles=noise_legend, title="noise", loc="upper left",
                     frameon=False, fontsize=10, title_fontsize=11)
    ax.add_artist(leg1)
    ax.legend(handles=model_legend, title="model", loc="lower right",
              frameon=False, fontsize=10, title_fontsize=11)

    ax.axhline(0, color=INK, linewidth=0.5, alpha=0.6)
    ax.axvline(0, color=INK, linewidth=0.5, alpha=0.6)
    ax.set_xlabel(r"$\Delta$ token-overlap per component (LOO)")
    ax.set_ylabel(r"$\Delta$ F1 per component (LOO)")
    title_r = f"Pearson r = {r:.2f}" if np.isfinite(r) else "Pearson r = n/a"
    ax.set_title(f"Mechanism: token-set stability predicts NER F1 ({title_r}, n={len(x)})",
                 fontsize=12, pad=10)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return r


def main():
    story, comb, f1c, tokc = load()
    fig_fertility(story, os.path.join(OUT, "fertility_3panel.pdf"))
    print("wrote fertility_3panel.pdf")
    fig_overlap(story, os.path.join(OUT, "overlap_3panel.pdf"))
    print("wrote overlap_3panel.pdf")
    fig_f1_heatmap(comb, os.path.join(OUT, "f1_heatmap.pdf"))
    print("wrote f1_heatmap.pdf")
    fig_ablation_heatmap(f1c, os.path.join(OUT, "ablation_heatmap.pdf"))
    print("wrote ablation_heatmap.pdf")
    r = fig_mechanism(f1c, tokc, os.path.join(OUT, "mechanism_scatter.pdf"))
    print(f"wrote mechanism_scatter.pdf (pearson r = {r:.3f})")


if __name__ == "__main__":
    main()
