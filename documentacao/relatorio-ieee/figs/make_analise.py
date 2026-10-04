"""Figuras de analise a partir de um modelo treinado (sem novo treino).

Uso (no ambiente ref-gaussian):  python make_analise.py [cena]
Gera em figs/:
  <cena>_decomposicao.png  mapas intermediarios da renderizacao diferida (visualize/050000.png)
  <cena>_envmap.png        mapa de ambiente aprendido (visualize/050000_env.png)
  <cena>_psnr_vistas.png   PSNR por vista de teste (renders do eval.py vs. imagens reais)
  <cena>_extremos.png      melhor e pior vista com mapa de erro
e imprime as estatisticas usadas no texto.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
RG = os.path.abspath(os.path.join(AQUI, "..", "..", "..", "ref-gaussian"))


def fonte(tam):
    try:
        return ImageFont.truetype("arial.ttf", tam)
    except OSError:
        return ImageFont.load_default()


def rotular(paineis, titulos, tam=34, faixa=46):
    w, h = paineis[0].size
    out = Image.new("RGB", (w * len(paineis), h + faixa), "white")
    d = ImageDraw.Draw(out)
    f = fonte(tam)
    for i, (p, t) in enumerate(zip(paineis, titulos)):
        out.paste(p, (i * w, faixa))
        tw = d.textlength(t, font=f)
        d.text((i * w + (w - tw) / 2, 4), t, fill="black", font=f)
    return out


def decomposicao(cena, it=50000):
    """Recorta paineis da grade salva por save_training_vis (make_grid nrow=4, padding=2, 800px)."""
    g = Image.open(os.path.join(RG, "output", cena, "visualize", f"{it:06d}.png")).convert("RGB")
    escala = (3 * 800 + 4 * 2) / g.size[1]
    lado = 800 / escala

    def painel(idx):
        r, c = divmod(idx, 4)
        x0, y0 = (2 + c * 802) / escala, (2 + r * 802) / escala
        return g.crop((round(x0), round(y0), round(x0 + lado), round(y0 + lado))).resize((400, 400))

    # ordem da grade (iteracao > volume_render_until_iter): 0 GT, 1 render, 2 albedo, 3 difuso,
    # 4 especular, 5 intensidade de reflexao, 6 rugosidade, 7 alpha, 8 profundidade,
    # 9 normal renderizada, 10 normal da superficie, 11 erro
    idx = [2, 3, 4, 5, 6, 11]
    tit = ["Albedo", "Difuso", "Especular", "Metalicidade", "Rugosidade", "Erro |GT - render|"]
    rotular([painel(i) for i in idx], tit).save(os.path.join(AQUI, f"{cena}_decomposicao.png"))


def envmap(cena, it=50000):
    e = Image.open(os.path.join(RG, "output", cena, "visualize", f"{it:06d}_env.png")).convert("RGB")
    w, h = e.size
    # dois paineis empilhados (env1, env2) com padding 2 de make_grid; usa o segundo (orientacao natural)
    baixo = e.crop((0, h // 2, w, h))
    baixo.save(os.path.join(AQUI, f"{cena}_envmap.png"))


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(1.0 / mse)


def sobre_branco(p):
    im = np.asarray(Image.open(p).convert("RGBA"), dtype=np.float32) / 255.0
    return im[..., :3] * im[..., 3:4] + (1 - im[..., 3:4])


def por_vista(cena):
    dados = os.path.join(RG, "data", "ref_nerf", cena)
    frames = json.load(open(os.path.join(dados, "transforms_test.json")))["frames"]
    rend = os.path.join(RG, "output", cena, "test", "renders", "rgb")
    vals, imgs = [], []
    for i, fr in enumerate(frames):
        gt = sobre_branco(os.path.join(dados, fr["file_path"]) + ".png")
        rd = np.asarray(Image.open(os.path.join(rend, f"{i:05d}.png")).convert("RGB"), dtype=np.float32) / 255.0
        vals.append(psnr(rd, gt))
        imgs.append((gt, rd))
    vals = np.array(vals)

    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.7), dpi=200, gridspec_kw={"width_ratios": [1.6, 1]})
    ax[0].plot(vals, lw=0.8, color="tab:orange")
    ax[0].axhline(vals.mean(), color="gray", ls="--", lw=0.7)
    ax[0].set_xlabel("vista de teste", fontsize=6)
    ax[0].set_ylabel("PSNR (dB)", fontsize=6)
    ax[1].hist(vals, bins=20, color="tab:orange", alpha=0.85)
    ax[1].set_xlabel("PSNR (dB)", fontsize=6)
    ax[1].set_ylabel("nº de vistas", fontsize=6)
    for a in ax:
        a.tick_params(labelsize=5)
        a.grid(alpha=0.3, lw=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(AQUI, f"{cena}_psnr_vistas.png"))
    plt.close()

    imin, imax = int(vals.argmin()), int(vals.argmax())
    paineis, tit = [], []
    for nome, i in (("pior", imin), ("melhor", imax)):
        gt, rd = imgs[i]
        err = np.clip(np.abs(gt - rd).mean(-1) * 4, 0, 1)  # erro absoluto medio amplificado 4x
        mapa = (plt.cm.inferno(err)[..., :3] * 255).astype(np.uint8)
        for arr, t in ((gt, f"Real (vista {i})"), (rd, f"Render - {vals[i]:.1f} dB"), (mapa, "Erro (4x)")):
            a = arr if arr.dtype == np.uint8 else (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            paineis.append(Image.fromarray(a).resize((400, 400)))
            tit.append(t)
    rotular(paineis, tit, tam=28).save(os.path.join(AQUI, f"{cena}_extremos.png"))

    print(f"{cena}: {len(vals)} vistas | media {vals.mean():.2f} dB | desvio {vals.std():.2f} | "
          f"min {vals.min():.2f} (vista {imin}) | max {vals.max():.2f} (vista {imax}) | "
          f"mediana {np.median(vals):.2f} | vistas < 25 dB: {(vals < 25).sum()}")


if __name__ == "__main__":
    cena = sys.argv[1] if len(sys.argv) > 1 else "toaster"
    decomposicao(cena)
    envmap(cena)
    por_vista(cena)
