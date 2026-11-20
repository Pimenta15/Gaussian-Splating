"""Avalia por vista as renderizações do Ref-Gaussian (eval.py --save_images) com os mesmos grupos de elevação.

    python refgs_avaliar.py /dados/banda/car_b25 /exp/saidas/car/refgs_b25 25
"""
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from vitrine_worker.metrics import LPIPS, psnr, ssim


def main():
    dados, modelo, banda = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    meta = json.loads((dados / "transforms_test.json").read_text())
    lp = LPIPS("cuda")
    linhas = []
    for idx, fr in enumerate(meta["frames"]):
        gt = cv2.imread(str(dados / (fr["file_path"] + ".png")), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        if gt.ndim == 3 and gt.shape[2] == 4:
            gt = gt[..., :3] * gt[..., 3:4] + (1 - gt[..., 3:4])  # sobre fundo branco, como no treino
        else:  # ball: a imagem já vem em RGB sobre fundo branco
            gt = gt[..., :3]
        gt = gt[..., ::-1].copy()
        pr = cv2.imread(str(modelo / "test" / "renders" / "rgb" / f"{idx:05d}.png"))[..., ::-1].astype(np.float32) / 255
        a = torch.from_numpy(pr.copy()).cuda().permute(2, 0, 1)[None]
        b = torch.from_numpy(gt).cuda().permute(2, 0, 1)[None]
        t = [r[3] for r in fr["transform_matrix"][:3]]
        e = math.degrees(math.asin(t[2] / math.sqrt(sum(x * x for x in t))))
        linhas.append({"nome": Path(fr["file_path"]).name, "origem": fr["origem"], "elev": e,
                       "refgs": {"psnr": psnr(a, b), "ssim": float(ssim(a, b)), "lpips": lp(a, b)}})
    (modelo / "por_vista.json").write_text(json.dumps(linhas, indent=1))
    # mesma vista de cima usada nas figuras do 3DGS
    figs = Path(str(modelo).replace("refgs_b", "b") + "_sh3") / "figs" / "vistas.json"
    if figs.exists():
        vistas = json.loads(figs.read_text())
        for rot, v in vistas.items():
            for idx, fr in enumerate(meta["frames"]):
                if Path(fr["file_path"]).name == v["nome"] and fr["origem"] == v["origem"]:
                    shutil.copy(modelo / "test" / "renders" / "rgb" / f"{idx:05d}.png", figs.parent / f"{rot}_refgs.png")
    for g, f in (("faixa", lambda ln: ln["origem"] == "teste" and ln["elev"] < banda),
                 ("acima", lambda ln: ln["origem"] == "teste" and ln["elev"] >= banda),
                 ("cima", lambda ln: ln["origem"] != "teste" and ln["elev"] >= 50)):
        sel = [ln["refgs"]["psnr"] for ln in linhas if f(ln)]
        print(f"{g}: n={len(sel)} psnr={np.mean(sel):.2f}", flush=True)


if __name__ == "__main__":
    main()
