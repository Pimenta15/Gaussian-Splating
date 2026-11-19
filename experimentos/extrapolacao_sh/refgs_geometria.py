"""Erro das normais do Ref-Gaussian por grupo de elevação, contra as normais de referência do Shiny Blender.

Usa os mapas de normal que o eval.py do Ref-Gaussian salva (test/renders/normal, rend_normal * 0,5 + 0,5, já no
referencial do mundo). São as normais que o próprio modelo usa para calcular o reflexo. Para o 3DGS, geometria.py
deriva as normais da profundidade (o 3DGS não tem normal própria). Mesma máscara de interior nos dois.

    python refgs_geometria.py <dados_banda/car_b25> <ref_nerf> car <saidas/car/refgs_b25> 25 [saida.json]
Também serve para o Ref-Gaussian treinado com o hemisfério todo: dados = ref_nerf/car (sem campo origem).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def ler(caminho: Path) -> np.ndarray:
    return np.asarray(Image.open(caminho)).astype(np.float32)


def main():
    banda_dir, ref_nerf, cena, modelo, banda = (Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]),
                                                 float(sys.argv[5]))
    meta = json.loads((banda_dir / "transforms_test.json").read_text())
    vistas_f = modelo.parent / f"b{banda:g}_sh3" / "figs" / "vistas.json"
    vistas = json.loads(vistas_f.read_text()) if vistas_f.exists() else {}
    linhas = []
    for idx, fr in enumerate(meta["frames"]):
        nome, origem = Path(fr["file_path"]).name, fr.get("origem", "teste")
        t = [r[3] for r in fr["transform_matrix"][:3]]
        e = math.degrees(math.asin(t[2] / math.sqrt(sum(x * x for x in t))))
        if origem == "teste":
            g = "faixa" if e < banda else "acima"
        elif e >= 50:
            g = "cima"
        else:
            continue
        pasta = ref_nerf / cena / ("test" if origem == "teste" else "train")
        img = ler(pasta / f"{nome}.png")
        a = img[..., 3] if img.ndim == 3 and img.shape[2] == 4 else ler(pasta / f"{nome}_alpha.png")
        if a.ndim == 3:
            a = a[..., 0]
        obj_ref = torch.from_numpy(a / 255 > 0.5)
        n_ref = torch.from_numpy(ler(pasta / f"{nome}_normal.png")[..., :3] / 255 * 2 - 1)
        n_ref = n_ref / n_ref.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        n = torch.from_numpy(ler(modelo / "test" / "renders" / "normal" / f"{idx:05d}.png")[..., :3] / 255 * 2 - 1)
        mag = n.norm(dim=-1)  # rend_normal é a soma ponderada pela opacidade: ~0 no fundo
        n = n / mag[..., None].clamp_min(1e-6)
        obj = mag > 0.5
        interior = F.max_pool2d((~obj_ref).float()[None, None], 11, 1, 5)[0, 0] < 0.5
        borda = torch.zeros_like(obj)
        borda[3:-3, 3:-3] = True
        msk = obj & obj_ref & interior & borda
        cos = (n * n_ref).sum(-1).clamp(-1, 1)
        ang = torch.rad2deg(torch.arccos(cos))[msk]
        linhas.append({"nome": nome, "origem": origem, "elev": e, "grupo": g,
                       "normal_erro_medio": float(ang.mean()), "normal_erro_mediana": float(ang.median()),
                       "normal_frac_15": float((ang < 15).float().mean()),
                       "frac_invertida": float((cos[msk] < 0).float().mean())})
        for rotulo, v in vistas.items():
            if v["nome"] == nome and v["origem"].startswith(origem[:5]):
                vis = ((n * 0.5 + 0.5).clamp(0, 1) * 255).byte().numpy().copy()
                vis[~obj.numpy()] = 255
                Image.fromarray(vis).save(vistas_f.parent / f"{rotulo}_normal_refgs.png")
    resumo = {}
    for g in ("faixa", "acima", "cima"):
        sel = [ln for ln in linhas if ln["grupo"] == g]
        resumo[g] = {"n": len(sel), **{k: float(np.mean([ln[k] for ln in sel])) for k in
                                       ("normal_erro_medio", "normal_erro_mediana", "normal_frac_15", "frac_invertida")}}
    print(cena, "refgs", json.dumps({g: {k: round(v, 3) for k, v in d.items()} for g, d in resumo.items()}), flush=True)
    saida = Path(sys.argv[6]) if len(sys.argv) > 6 else modelo / "geometria.json"
    saida.write_text(json.dumps({"resumo": resumo, "por_vista": linhas}, indent=1))


if __name__ == "__main__":
    main()
