"""Figura analítica: quanto dos 16 coeficientes de SH (grau 3) uma faixa de elevação consegue determinar.

Autovalores da matriz de Gram G = B_D^T B_D / |D| das direções observadas D. Autovalor ~0 = combinação
de coeficientes que não muda a cor em nenhuma direção filmada: o treino não tem como fixá-la, e ela
aparece livremente nas vistas de fora da faixa.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, r"C:\Users\Pimenta\Desktop\vitrine3d\services\worker")
from vitrine_worker.sh_observado import base_sh, direcoes_observadas  # noqa: E402

FAIXAS = [("Porsche real (0,4°–3,8°)", 0.4, 3.8), ("7°–15°", 7, 15), ("7°–25°", 7, 25),
          ("7°–89° (hemisfério)", 7, 89), ("esfera inteira", -89.5, 89.5)]
CORES = ["#c0392b", "#e67e22", "#2e86c1", "#27ae60", "#7f8c8d"]


def espectro(emin, emax, grau=3):
    d = direcoes_observadas(emin, emax, margem=0, n_az=180, n_el=60)
    # amostragem uniforme em área: peso cos(elevação)
    el = np.arcsin(-d[:, 2])
    w = np.cos(el)
    B = base_sh(d, grau)
    G = (B * w[:, None]).T @ B / w.sum()
    return np.sort(np.linalg.eigvalsh(G))[::-1]


def main(saida: Path):
    fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=200)
    linhas = []
    for (nome, a, b), cor in zip(FAIXAS, CORES):
        ev = espectro(a, b)
        ev_n = np.clip(ev / ev.max(), 1e-12, None)
        ax.semilogy(np.arange(1, 17), ev_n, "o-", ms=2.5, lw=1.1, color=cor, label=nome)
        linhas.append((nome, int((ev_n < 1e-2).sum()), int((ev_n < 1e-4).sum())))
    ax.axhline(1e-2, color="k", lw=0.6, ls=":")
    ax.set_xlabel("autovalor (ordem decrescente)", fontsize=7)
    ax.set_ylabel("autovalor / maior", fontsize=7)
    ax.set_ylim(1e-9, 2)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, frameon=False, loc="lower left")
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    saida.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida / "espectro_gram.pdf")
    fig.savefig(saida / "espectro_gram.png")
    for nome, n2, n4 in linhas:
        print(f"{nome:28s} autovalores < 1e-2: {n2:2d}   < 1e-4: {n4:2d}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "figs")
