"""Figuras grandes para os slides (fontes legíveis em 1920x1080), a partir dos mesmos resultados das tabelas.

Uso: python -I fig_slides.py <saidas> <destino>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CENAS = ["ball", "car", "coffee", "helmet", "teapot", "toaster"]
ESCURO, LARANJA, CINZA, AZUL = "#14213D", "#C25E12", "#8A96A3", "#2E6FA8"
TINTA, TINTA2 = "#1F2937", "#4A5568"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.edgecolor": "#B8BEC6",
                     "axes.labelcolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2})


def carregar(saidas: Path, rotulo: str) -> dict:
    res = {}
    for c in CENAS:
        f = saidas / c / rotulo / "resultado.json"
        if f.exists():
            res[c] = json.loads(f.read_text())
    return res


def serie(res: dict, var: str, bins: np.ndarray) -> tuple[list, list]:
    xs, ys = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        por_cena = [np.mean(s) for r in res.values()
                    if (s := [ln[var]["psnr"] for ln in r["por_vista"] if a <= ln["elev"] < b and var in ln])]
        if len(por_cena) == len(res):
            xs.append((a + b) / 2)
            ys.append(float(np.mean(por_cena)))
    return xs, ys


def eixo(ax, banda: float) -> None:
    ax.axvspan(0, banda, color=AZUL, alpha=0.10, lw=0)
    ax.text(banda / 2, 0.04, "vistas\nfilmadas", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=18, color=AZUL)
    ax.set_xlim(5, 90)
    ax.set_xticks([10, 25, 50, 70, 89])
    ax.set_xticklabels(["10°", "25°", "50°", "70°", "89°"])
    ax.set_xlabel("elevação da câmera de avaliação")
    ax.set_ylabel("PSNR médio (dB)")
    ax.grid(axis="y", alpha=0.35, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def perda(res: dict, banda: float, destino: Path, ref: dict | None) -> None:
    bins = np.arange(5, 95, 5)
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    xs, ys = serie(res, "original", bins)
    ax.plot(xs, ys, "-", color=ESCURO, lw=3, marker="o", ms=7, label="treino só abaixo de 25°")
    if ref:
        xr, yr = serie(ref, "original", bins)
        ax.plot(xr, yr, "--", color=LARANJA, lw=3, label="treino com todas as alturas")
        ax.legend(frameon=False, fontsize=18, loc="upper right")
    eixo(ax, banda)
    fig.tight_layout()
    fig.savefig(destino / "slide_perda.png")
    plt.close(fig)


def correcao(res: dict, banda: float, destino: Path) -> None:
    bins = np.arange(5, 95, 5)
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    estilos = [("original", "3DGS original", ESCURO, "-", 3), ("visib", "com a projeção", LARANJA, "--", 3),
               ("dc", "sem reflexo (só cor média)", CINZA, ":", 3)]
    pontas = {}
    for v, nome, cor, ls, lw in estilos:
        xs, ys = serie(res, v, bins)
        ax.plot(xs, ys, ls, color=cor, lw=lw, label=nome)
        pontas[v] = (xs, ys)
    ax.legend(frameon=False, fontsize=18, loc="upper right")
    eixo(ax, banda)
    fig.tight_layout()
    fig.savefig(destino / "slide_correcao.png")
    plt.close(fig)


def main() -> None:
    saidas, destino = Path(sys.argv[1]), Path(sys.argv[2])
    destino.mkdir(parents=True, exist_ok=True)
    res = carregar(saidas, "b25_sh3")
    ref = carregar(saidas, "completo_sh3")
    perda(res, 25, destino, ref if len(ref) == len(res) else None)
    correcao(res, 25, destino)


if __name__ == "__main__":
    main()
