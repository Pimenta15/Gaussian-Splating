"""Resumo 3DGS x Ref-Gaussian nas cenas em que o Ref-Gaussian foi treinado na faixa.

PSNR por grupo (faixa, acima, de cima) e erro das normais, para os dois modelos treinados só abaixo da faixa, e o
PSNR do Ref-Gaussian treinado com o hemisfério todo (replicação do artigo, ref-gaussian/output/<cena>) nas vistas
de teste, como referência do que o mesmo modelo alcança quando vê as alturas.

    python resumo_refgs.py saidas ../../ref-gaussian [saida.json]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

GRUPOS = ("faixa", "acima", "cima")


def grupo(ln: dict, banda: float = 25) -> str | None:
    if ln["origem"] == "teste":
        return "faixa" if ln["elev"] < banda else "acima"
    return "cima" if ln["elev"] >= 50 else None


def psnr_completo(refgs: Path, cena: str) -> dict:
    """PSNR por grupo das renderizações de teste do Ref-Gaussian treinado com o hemisfério todo."""
    d = refgs / "data" / "ref_nerf" / cena
    meta = json.loads((d / "transforms_test.json").read_text())
    res = {"faixa": [], "acima": []}
    for i, fr in enumerate(meta["frames"]):
        t = [r[3] for r in fr["transform_matrix"][:3]]
        e = math.degrees(math.asin(t[2] / math.sqrt(sum(x * x for x in t))))
        img = np.asarray(Image.open(d / "test" / f"{Path(fr['file_path']).name}.png")).astype(np.float32) / 255
        gt = img[..., :3] * img[..., 3:4] + (1 - img[..., 3:4]) if img.shape[2] == 4 else img[..., :3]
        pr = np.asarray(Image.open(refgs / "output" / cena / "test" / "renders" / "rgb" / f"{i:05d}.png"))
        pr = pr[..., :3].astype(np.float32) / 255
        res["faixa" if e < 25 else "acima"].append(-10 * math.log10(((pr - gt) ** 2).mean()))
    return {g: float(np.mean(v)) for g, v in res.items()}


def main():
    saidas, refgs = Path(sys.argv[1]), Path(sys.argv[2])
    cenas = [c.name for c in sorted(saidas.iterdir()) if (c / "refgs_b25" / "por_vista.json").exists()
             and (c / "refgs_b25" / "geometria.json").exists()]
    out = {"cenas": cenas, "por_cena": {}}
    for c in cenas:
        rg = json.loads((saidas / c / "refgs_b25" / "por_vista.json").read_text())
        gs = json.loads((saidas / c / "b25_sh3" / "resultado.json").read_text())["por_vista"]
        d = {"psnr_3dgs": {}, "psnr_refgs": {}}
        for g in GRUPOS:
            d["psnr_refgs"][g] = float(np.mean([ln["refgs"]["psnr"] for ln in rg if grupo(ln) == g]))
            d["psnr_3dgs"][g] = float(np.mean([ln["original"]["psnr"] for ln in gs if grupo(ln) == g]))
        nr = json.loads((saidas / c / "refgs_b25" / "geometria.json").read_text())["resumo"]
        ng = json.loads((saidas / c / "b25_sh3" / "geometria.json").read_text())["resumo"]
        d["normal_refgs"] = {g: nr[g]["normal_erro_medio"] for g in GRUPOS}
        d["normal_3dgs_prof"] = {g: ng[g]["normal_erro_medio"] for g in GRUPOS}
        d["normal_3dgs_eixo"] = {g: ng[g]["eixo_erro_medio"] for g in GRUPOS}
        d["psnr_refgs_completo"] = psnr_completo(refgs, c)
        out["por_cena"][c] = d
    media = {}
    for k in ("psnr_3dgs", "psnr_refgs", "normal_refgs", "normal_3dgs_prof", "normal_3dgs_eixo", "psnr_refgs_completo"):
        gs_ = GRUPOS if k != "psnr_refgs_completo" else ("faixa", "acima")
        media[k] = {g: float(np.mean([out["por_cena"][c][k][g] for c in cenas])) for g in gs_}
    for k in ("psnr_3dgs", "psnr_refgs"):
        media[k]["queda_cima"] = media[k]["faixa"] - media[k]["cima"]
    out["media"] = media
    print(json.dumps({"cenas": cenas, "media": {k: {g: round(v, 2) for g, v in d.items()} for k, d in media.items()}},
                     indent=1, ensure_ascii=False))
    for c in cenas:
        print(c, {k: {g: round(v, 2) for g, v in d.items()} for k, d in out["por_cena"][c].items()})
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
