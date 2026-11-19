"""Figura do protocolo: posição (azimute x elevação) das câmeras de treino e dos três grupos de avaliação."""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pos(fr):
    t = [r[3] for r in fr["transform_matrix"][:3]]
    n = math.sqrt(sum(x * x for x in t))
    return math.degrees(math.atan2(t[1], t[0])), math.degrees(math.asin(t[2] / n))


def main(ref: Path, destino: Path, cena="car", banda=25.0):
    tr = [pos(f) for f in json.loads((ref / cena / "transforms_train.json").read_text())["frames"]]
    te = [pos(f) for f in json.loads((ref / cena / "transforms_test.json").read_text())["frames"]]
    grupos = [
        ("treino (abaixo de 25°)", [p for p in tr if p[1] < banda], "#2e86c1", "o", 14),
        ("teste dentro da faixa", [p for p in te if p[1] < banda], "#7fb3d5", ".", 6),
        ("teste acima (25°–47°)", [p for p in te if p[1] >= banda], "#e67e22", ".", 6),
        ("de cima (treino não usado, ≥50°)", [p for p in tr if p[1] >= 50], "#c0392b", "^", 14),
    ]
    fig, ax = plt.subplots(figsize=(3.5, 1.9), dpi=250)
    ax.axhspan(0, banda, color="#2e86c1", alpha=0.07, lw=0)
    for nome, ps, cor, m, s in grupos:
        ax.scatter([p[0] for p in ps], [p[1] for p in ps], s=s, c=cor, marker=m, label=f"{nome} ({len(ps)})",
                   edgecolors="none")
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 92)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([0, 25, 50, 75, 90])
    ax.set_xlabel("azimute (graus)", fontsize=7)
    ax.set_ylabel("elevação (graus)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.2, lw=0.4)
    ax.legend(fontsize=5, frameon=True, framealpha=0.9, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2,
              handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(destino / "protocolo.pdf", bbox_inches="tight")
    fig.savefig(destino / "protocolo.png", bbox_inches="tight")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
