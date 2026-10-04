"""Gera as figuras qualitativas do relatorio a partir dos renders do eval.py.

Uso (no ambiente ref-gaussian):  python make_figs.py [cena] [indice_vista]
"""
import os
import sys
import json

from PIL import Image, ImageDraw, ImageFont

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RG = os.path.join(RAIZ, "ref-gaussian")


def sobre_branco(img):
    """Compoe imagens RGBA do dataset sobre fundo branco (--white_background)."""
    img = img.convert("RGBA")
    fundo = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(fundo, img).convert("RGB")


def painel(cena, idx, destino):
    dados = os.path.join(RG, "data", "ref_nerf", cena)
    with open(os.path.join(dados, "transforms_test.json")) as f:
        frame = json.load(f)["frames"][idx]["file_path"]  # ex.: ./test/r_0
    base = os.path.join(dados, frame)
    renders = os.path.join(RG, "output", cena, "test", "renders")

    imgs = [
        ("Real (GT)", sobre_branco(Image.open(base + ".png"))),
        ("Ref-Gaussian", Image.open(os.path.join(renders, "rgb", f"{idx:05d}.png")).convert("RGB")),
        ("Normal real", sobre_branco(Image.open(base + "_normal.png"))),
        ("Normal estimada", Image.open(os.path.join(renders, "normal", f"{idx:05d}.png")).convert("RGB")),
    ]
    w, h = imgs[1][1].size
    imgs = [(t, im.resize((w, h))) for t, im in imgs]

    faixa = 70
    out = Image.new("RGB", (w * len(imgs), h + faixa), "white")
    draw = ImageDraw.Draw(out)
    try:
        fonte = ImageFont.truetype("arial.ttf", 52)
    except OSError:
        fonte = ImageFont.load_default()
    for i, (titulo, im) in enumerate(imgs):
        out.paste(im, (i * w, faixa))
        tw = draw.textlength(titulo, font=fonte)
        draw.text((i * w + (w - tw) / 2, 4), titulo, fill="black", font=fonte)
    out.save(destino)
    print("salvo:", destino, out.size)


if __name__ == "__main__":
    cena = sys.argv[1] if len(sys.argv) > 1 else "toaster"
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    painel(cena, idx, os.path.join(os.path.dirname(__file__), f"{cena}_qualitativo.png"))
