"""Figura 1 do artigo: Porsche 911 da Vitrine 3D, na altura filmada e de cima, antes e depois da projeção.

Uso: python -I fig_teaser.py <quadro.jpg> <saidas/porsche> <destino figs/> [elevação de cima, padrão 70]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def recorta_conteudo(img: np.ndarray, margem: float = 0.04) -> np.ndarray:
    """Recorta a caixa do carro (pixels que diferem do fundo do canto) com uma margem."""
    fundo = img[2, 2].astype(np.float32)
    dif = np.abs(img.astype(np.float32) - fundo).sum(axis=2) > 0.08 * (img.max() if img.max() > 1 else 1)
    ys, xs = np.nonzero(dif)
    if len(xs) == 0:
        return img
    h, w = img.shape[:2]
    my, mx = int(margem * h), int(margem * w)
    return img[max(ys.min() - my, 0):min(ys.max() + my, h), max(xs.min() - mx, 0):min(xs.max() + mx, w)]


def main() -> None:
    quadro, pasta, destino = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    el = int(sys.argv[4]) if len(sys.argv) > 4 else 70
    q = plt.imread(quadro).astype(np.float32) / 255
    mascara = quadro.parent.parent / "mascaras" / quadro.name  # máscara do carro gerada pelo próprio pipeline
    if mascara.exists():  # só o carro, sem o público do salão ao fundo
        m = plt.imread(mascara).astype(np.float32) / 255
        m = (m[..., 0] if m.ndim == 3 else m)[..., None]
        q = q * m + (1 - m)
        q = recorta_conteudo(q)
    paineis = [(q, "(a) quadro do vídeo, carro segmentado"),
               (recorta_conteudo(plt.imread(pasta / "el02_depois.png")), "(b) modelo na altura filmada"),
               (recorta_conteudo(plt.imread(pasta / f"el{el:02d}_antes.png")), f"(c) de cima ({el}°), 3DGS"),
               (recorta_conteudo(plt.imread(pasta / f"el{el:02d}_depois.png")), f"(d) de cima ({el}°), com projeção")]
    fig, axs = plt.subplots(2, 2, figsize=(3.45, 2.6), dpi=300)
    for ax, (img, titulo) in zip(axs.ravel(), paineis):
        ax.imshow(img)
        ax.set_title(titulo, fontsize=6, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.3)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.005, wspace=0.03, hspace=0.12)
    fig.savefig(destino / "teaser.pdf")
    fig.savefig(destino / "teaser.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
