"""Oráculos complementares para separar cor de geometria nas vistas de cima (treino não usado, elevação >= 50°).

O oráculo de cor (oraculo.py) reajusta todos os SH. Como a cor dependente de vista pode disfarçar erro de forma,
aqui há dois controles, ambos ajustados nas vistas de índice par e avaliados nas ímpares:
  - cor_dc: só a cor constante (sh0) é livre e os SH direcionais são zerados: uma cor por Gaussiana, igual
    de todos os ângulos, não consegue compensar forma errada vista a vista;
  - geometria: posições, escalas, rotações e opacidades livres, com a cor (SH) congelada como saiu do treino.
Também o oráculo de geometria ajustado num lado do objeto e avaliado no outro (mesmo controle do oraculo.py).

    python oraculo2.py --cena car --banda 25
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from oraculo import media_psnr
from rodar import azimute, elevacao
from vitrine_worker.datasets import ler_transforms, raio_da_cena
from vitrine_worker.metrics import ssim
from vitrine_worker.train_3dgs import _renderizar


def ajustar(params, vistas, livres: dict, zerar_shN: bool = False, passos: int = 3000, decai: tuple = ()):
    """Ajusta só os parâmetros em `livres` ({nome: lr}); os demais ficam congelados."""
    p = {k: v.clone() for k, v in params.items()}
    if zerar_shN:
        p["shN"] = torch.zeros_like(p["shN"])
    vars_ = {k: torch.nn.Parameter(p[k].clone()) for k in livres}
    opt = {k: torch.optim.Adam([vars_[k]], lr=lr, eps=1e-15) for k, lr in livres.items()}
    agenda = [torch.optim.lr_scheduler.ExponentialLR(opt[k], gamma=0.01 ** (1.0 / passos)) for k in decai]
    grau = int(round(math.sqrt(p["shN"].shape[1] + 1))) - 1
    imgs = [torch.from_numpy(c.imagem).cuda() for c in vistas]
    Ks = [torch.from_numpy(c.K).float().cuda()[None] for c in vistas]
    vms = [torch.from_numpy(c.viewmat).float().cuda()[None] for c in vistas]
    fundo = torch.ones(1, 3, device="cuda")
    for passo in range(passos):
        i = passo % len(vistas)
        q = dict(p, **vars_)
        rgb, _, _ = _renderizar(q, Ks[i], vms[i], vistas[i].largura, vistas[i].altura, grau, fundo)
        pred = rgb[0]
        perda = 0.8 * F.l1_loss(pred, imgs[i]) + 0.2 * (1 - ssim(pred.permute(2, 0, 1)[None],
                                                                 imgs[i].permute(2, 0, 1)[None]))
        for o in opt.values():
            o.zero_grad(set_to_none=True)
        perda.backward()
        for o in opt.values():
            o.step()
        for a in agenda:
            a.step()
    return dict(p, **{k: v.detach() for k, v in vars_.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="/data")
    ap.add_argument("--cena", required=True)
    ap.add_argument("--banda", type=float, default=25.0)
    ap.add_argument("--saida", default="/exp/saidas")
    ap.add_argument("--passos", type=int, default=3000)
    a = ap.parse_args()
    pasta = Path(a.saida) / a.cena / f"b{a.banda:g}_sh3"
    params = {k: v.cuda() for k, v in torch.load(pasta / "params.pt").items()}
    treino = ler_transforms(Path(a.dados) / a.cena / "transforms_train.json")
    raio = raio_da_cena([c for c in treino if elevacao(c) < a.banda])
    vs = [c for c in treino if elevacao(c) >= 50]
    par, impar = vs[0::2], vs[1::2]
    lado_a = [c for c in vs if azimute(c) < 0]
    lado_b = [c for c in vs if azimute(c) >= 0]
    geo = {"means": 1.6e-4 * raio, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2}

    torch.manual_seed(0)
    so_dc = dict(params, shN=torch.zeros_like(params["shN"]))
    cor_dc = ajustar(params, par, {"sh0": 5e-3}, zerar_shN=True, passos=a.passos)
    geo_cruz = ajustar(params, par, geo, passos=a.passos, decai=("means",))
    geo_op = ajustar(params, lado_a, geo, passos=a.passos, decai=("means",))
    res = {"cima": {"n": len(vs), "n_impares": len(impar), "n_lado_b": len(lado_b),
                    "original_impares": media_psnr(params, impar),
                    "so_dc_sem_ajuste_impares": media_psnr(so_dc, impar),
                    "oraculo_cor_dc_impares": media_psnr(cor_dc, impar),
                    "oraculo_geometria_impares": media_psnr(geo_cruz, impar),
                    "original_lado_b": media_psnr(params, lado_b),
                    "oraculo_geometria_lado_oposto": media_psnr(geo_op, lado_b)}}
    print(a.cena, {k: round(v, 2) for k, v in res["cima"].items()}, flush=True)
    (pasta / ("oraculo2.json" if a.passos == 3000 else "oraculo2_teste.json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
