"""Oráculo de cor: quanto do erro nas vistas fora da faixa é de cor e quanto é de geometria.

Congela tudo menos os SH do modelo treinado na faixa e os ajusta DIRETO nas imagens de avaliação
(trapaça proposital). O PSNR resultante é o teto que qualquer correção só de cor poderia atingir com
essa geometria. Duas versões:
  - todas: ajusta e avalia nas mesmas vistas (teto absoluto);
  - cruzada: ajusta nas vistas de índice par e avalia nas ímpares (teto de uma cor que generaliza).

    python oraculo.py --cena car --banda 25
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from rodar import azimute, elevacao, renderizar
from vitrine_worker.datasets import ler_transforms
from vitrine_worker.metrics import psnr, ssim
from vitrine_worker.train_3dgs import _renderizar


def ajustar_cor(params, vistas, passos=3000):
    p = {k: v.clone() for k, v in params.items()}
    sh0 = torch.nn.Parameter(p["sh0"].clone())
    shN = torch.nn.Parameter(p["shN"].clone())
    opt = torch.optim.Adam([{"params": [sh0], "lr": 5e-3}, {"params": [shN], "lr": 5e-4}], eps=1e-15)
    grau = int(round(math.sqrt(p["shN"].shape[1] + 1))) - 1
    imgs = [torch.from_numpy(c.imagem).cuda() for c in vistas]
    Ks = [torch.from_numpy(c.K).float().cuda()[None] for c in vistas]
    vms = [torch.from_numpy(c.viewmat).float().cuda()[None] for c in vistas]
    fundo = torch.ones(1, 3, device="cuda")
    for passo in range(passos):
        i = passo % len(vistas)
        q = dict(p, sh0=sh0, shN=shN)
        rgb, _, _ = _renderizar(q, Ks[i], vms[i], vistas[i].largura, vistas[i].altura, grau, fundo)
        pred = rgb[0]
        perda = 0.8 * F.l1_loss(pred, imgs[i]) + 0.2 * (1 - ssim(pred.permute(2, 0, 1)[None],
                                                                 imgs[i].permute(2, 0, 1)[None]))
        opt.zero_grad(set_to_none=True)
        perda.backward()
        opt.step()
    return dict(p, sh0=sh0.detach(), shN=shN.detach())


@torch.no_grad()
def media_psnr(params, vistas):
    return float(np.mean([psnr(renderizar(params, c), torch.from_numpy(c.imagem).cuda()) for c in vistas]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="/data")
    ap.add_argument("--cena", required=True)
    ap.add_argument("--banda", type=float, default=25.0)
    ap.add_argument("--saida", default="/exp/saidas")
    a = ap.parse_args()
    pasta = Path(a.saida) / a.cena / f"b{a.banda:g}_sh3"
    params = {k: v.cuda() for k, v in torch.load(pasta / "params.pt").items()}
    treino = ler_transforms(Path(a.dados) / a.cena / "transforms_train.json")
    teste = ler_transforms(Path(a.dados) / a.cena / "transforms_test.json")
    grupos = {"cima": [c for c in treino if elevacao(c) >= 50],
              "acima": [c for c in teste if a.banda <= elevacao(c)]}
    res = {}
    for g, vs in grupos.items():
        torch.manual_seed(0)
        tudo = ajustar_cor(params, vs)
        par, impar = vs[0::2], vs[1::2]
        cruz = ajustar_cor(params, par)
        # controle mais duro: ajusta num lado do objeto (azimute < 0) e avalia no lado oposto, para a cor
        # dependente de vista não poder "disfarçar" erro de geometria em vistas vizinhas às do ajuste
        lado_a = [c for c in vs if azimute(c) < 0]
        lado_b = [c for c in vs if azimute(c) >= 0]
        oposto = ajustar_cor(params, lado_a)
        res[g] = {"n": len(vs), "original": media_psnr(params, vs), "oraculo_todas": media_psnr(tudo, vs),
                  "original_impares": media_psnr(params, impar), "oraculo_cruzado_impares": media_psnr(cruz, impar),
                  "n_lado_b": len(lado_b), "original_lado_b": media_psnr(params, lado_b),
                  "oraculo_lado_oposto": media_psnr(oposto, lado_b)}
        print(a.cena, g, {k: round(v, 2) for k, v in res[g].items()}, flush=True)
        # imagem da vista de cima das figuras com a cor do oráculo ajustado no lado oposto (ou em todas)
        meta = pasta / "figs" / "vistas.json"
        if g == "cima" and meta.exists():
            v = json.loads(meta.read_text())["cima"]
            for c in vs:
                if c.nome == v["nome"]:
                    no_b = any(c is x for x in lado_b)  # identidade: Camera tem arrays, == não serve
                    img = renderizar(oposto if no_b else tudo, c).cpu().numpy()
                    cv2.imwrite(str(pasta / "figs" / "cima_oraculo.png"), (img[..., ::-1] * 255).astype(np.uint8))
                    res[g]["figura_oraculo"] = "lado_oposto" if no_b else "todas"
    (pasta / "oraculo.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
