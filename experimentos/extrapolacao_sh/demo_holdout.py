"""Renderiza o 3DGS treinado com todas as alturas sem as fotos de avaliação (holdout_sh3) nas vistas da varredura.

As vistas de treino da varredura estão todas entre as separadas (prepara_holdout.py), então o PSNR aqui é de vista
nunca vista. Grava explorar/<i>_holdout.jpg e acrescenta psnr_holdout a explorar/modelos.json.

    python demo_holdout.py --cena car
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from rodar import renderizar
from vitrine_worker.datasets import ler_transforms


def psnr_np(a, b):
    return float(-10 * math.log10(max(float(((a - b) ** 2).mean()), 1e-12)))


def main():
    cena = sys.argv[sys.argv.index("--cena") + 1]
    pasta = Path("/exp/saidas") / cena / "b25_sh3" / "explorar"
    f_mod = pasta / "modelos.json"
    modelos = json.loads(f_mod.read_text())
    meta = {m["i"]: m for m in json.loads((pasta / "meta.json").read_text())}
    p = Path("/exp/saidas") / cena / "holdout_sh3" / "params.pt"
    if not p.exists():
        print(cena, "sem modelo holdout")
        return
    params = {k: v.cuda() for k, v in torch.load(p).items()}
    dados = Path("/data") / cena
    cams = {("teste", c.nome): c for c in ler_transforms(dados / "transforms_test.json", fundo_branco=True)}
    cams.update({("treino_nao_usada", c.nome): c for c in ler_transforms(dados / "transforms_train.json",
                                                                          fundo_branco=True)})
    separadas = set(json.loads((Path("/exp/saidas") / cena / "holdout.json").read_text()))
    n = 0
    for m in modelos:
        origem = meta[m["i"]]["origem"]
        c = cams.get((origem, m["nome"]))
        if c is None:
            continue
        if origem != "teste" and m["nome"] not in separadas:
            continue  # vista de treino que não foi separada: não serve para mostrar extrapolação
        img = renderizar(params, c).cpu().numpy()
        m["psnr_holdout"] = psnr_np(img, c.imagem)
        arr = cv2.resize((img * 255).clip(0, 255).astype(np.uint8)[..., ::-1], (400, 400),
                         interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(pasta / f"{m['i']:02d}_holdout.jpg"), arr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        n += 1
    f_mod.write_text(json.dumps(modelos, indent=1))
    print(cena, n, "vistas renderizadas do 3DGS sem as fotos de avaliação", flush=True)


if __name__ == "__main__":
    main()
