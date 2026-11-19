"""Erro de geometria por grupo de elevação, medido contra a geometria de referência do Shiny Blender.

O Shiny Blender traz, para cada vista, o mapa de normais verdadeiro (r_X_normal.png, no referencial do mundo) e o
canal alfa da imagem (silhueta). Para o modelo treinado só abaixo da faixa, renderizamos a profundidade esperada,
derivamos as normais dela e comparamos nos mesmos três grupos da avaliação de cor:
  - erro angular médio das normais (graus) nos pixels do objeto;
  - IoU da silhueta (alfa renderizado > 0,5 contra alfa de referência > 0,5).
Se a geometria vista de cima for tão boa quanto a da faixa filmada, a queda de PSNR de cima é de cor.

    python geometria.py --cena car --banda 25
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from rodar import elevacao
from vitrine_worker.datasets import ler_transforms


@torch.no_grad()
def renderizar_geometria(params, c):
    from gsplat import rasterization  # noqa: PLC0415

    grau = int(round(math.sqrt(params["shN"].shape[1] + 1))) - 1
    out, alfa, _ = rasterization(
        means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]), colors=torch.cat([params["sh0"], params["shN"]], 1),
        viewmats=torch.from_numpy(c.viewmat).float().cuda()[None], Ks=torch.from_numpy(c.K).float().cuda()[None],
        width=c.largura, height=c.altura, sh_degree=grau, render_mode="RGB+ED", packed=False)
    return out[0, ..., 3], alfa[0, ..., 0]


@torch.no_grad()
def normal_eixo(params, c) -> torch.Tensor:
    """Normal renderizada do 3DGS: eixo mais curto de cada Gaussiana (virado para a câmera), composto por alfa.
    É a normal usada pelos métodos de reflexo; aqui, sem nenhuma suavização."""
    from gsplat import rasterization  # noqa: PLC0415

    w, x, y, z = F.normalize(params["quats"], dim=-1).unbind(-1)
    R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)
    k = params["scales"].argmin(-1)
    n = R[torch.arange(R.shape[0], device=R.device), :, k]
    cam = torch.from_numpy(c.c2w[:3, 3]).float().cuda()
    n = torch.where(((cam - params["means"]) * n).sum(-1, keepdim=True) < 0, -n, n)
    out, _, _ = rasterization(
        means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]), colors=n,
        viewmats=torch.from_numpy(c.viewmat).float().cuda()[None], Ks=torch.from_numpy(c.K).float().cuda()[None],
        width=c.largura, height=c.altura, sh_degree=None, render_mode="RGB", packed=False)
    m = out[0]
    return m / m.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def normais(z: torch.Tensor, m: torch.Tensor, K: np.ndarray, c2w: np.ndarray, suave: int = 9) -> torch.Tensor:
    """Normais no referencial do mundo a partir da profundidade z (câmera OpenCV), com suavização mascarada."""
    pad = suave // 2
    zm = F.avg_pool2d((z * m)[None, None], suave, 1, pad, count_include_pad=False)[0, 0]
    mm = F.avg_pool2d(m[None, None], suave, 1, pad, count_include_pad=False)[0, 0]
    zs = zm / mm.clamp_min(1e-6)
    h, w = z.shape
    v, u = torch.meshgrid(torch.arange(h, device=z.device, dtype=z.dtype),
                          torch.arange(w, device=z.device, dtype=z.dtype), indexing="ij")
    p = torch.stack([(u - K[0, 2]) / K[0, 0] * zs, (v - K[1, 2]) / K[1, 1] * zs, zs], -1)
    du = torch.zeros_like(p)
    dv = torch.zeros_like(p)
    du[:, 1:-1] = p[:, 2:] - p[:, :-2]
    dv[1:-1] = p[2:] - p[:-2]
    n = torch.cross(du, dv, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    n = torch.where(((n * p).sum(-1, keepdim=True) > 0), -n, n)  # aponta para a câmera
    R = torch.from_numpy(c2w[:3, :3]).to(z)
    return n @ R.T


def referencia(dados: Path, cena: str, origem: str, nome: str, h: int, w: int):
    pasta = dados / cena / ("test" if origem == "teste" else "train")
    img = cv2.imread(str(pasta / f"{nome}.png"), cv2.IMREAD_UNCHANGED)
    nrm = cv2.imread(str(pasta / f"{nome}_normal.png"), cv2.IMREAD_UNCHANGED)[..., :3][..., ::-1]
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[..., 3]
    else:  # ball: imagem RGB já sobre fundo branco, alfa num arquivo separado
        a = cv2.imread(str(pasta / f"{nome}_alpha.png"), cv2.IMREAD_GRAYSCALE)
    alfa = torch.from_numpy(a.astype(np.float32) / 255).cuda()
    n = torch.from_numpy(nrm.astype(np.float32) / 255 * 2 - 1).cuda()
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6), alfa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="/data")
    ap.add_argument("--cena", required=True)
    ap.add_argument("--banda", type=float, default=25.0)
    ap.add_argument("--rotulo", default=None, help="pasta do modelo (padrão b{banda}_sh3)")
    ap.add_argument("--saida", default="/exp/saidas")
    a = ap.parse_args()
    rot = a.rotulo or f"b{a.banda:g}_sh3"
    pasta = Path(a.saida) / a.cena / rot
    params = {k: v.cuda() for k, v in torch.load(pasta / "params.pt").items()}
    treino = ler_transforms(Path(a.dados) / a.cena / "transforms_train.json")
    teste = ler_transforms(Path(a.dados) / a.cena / "transforms_test.json")
    vistas = [("teste", c) for c in teste] + [("treino", c) for c in treino if elevacao(c) >= 50]
    meta = json.loads((pasta / "figs" / "vistas.json").read_text()) if (pasta / "figs" / "vistas.json").exists() else {}
    linhas = []
    for origem, c in vistas:
        e = elevacao(c)
        g = "cima" if origem == "treino" else ("faixa" if e < a.banda else "acima")
        z, alfa = renderizar_geometria(params, c)
        n_ref, alfa_ref = referencia(Path(a.dados), a.cena, origem, c.nome, c.altura, c.largura)
        obj, obj_ref = alfa > 0.5, alfa_ref > 0.5
        iou = float((obj & obj_ref).sum() / (obj | obj_ref).sum().clamp_min(1))
        n = normais(z, obj.float(), c.K, c.c2w)
        borda = torch.zeros_like(obj)
        borda[3:-3, 3:-3] = True
        # longe da silhueta: a suavização mistura fundo nos 2-3 pixels da borda
        interior = F.max_pool2d((~obj_ref).float()[None, None], 11, 1, 5)[0, 0] < 0.5
        msk = obj & obj_ref & borda & interior
        ang = torch.rad2deg(torch.arccos((n * n_ref).sum(-1).clamp(-1, 1)))[msk]
        n_e = normal_eixo(params, c)
        ang_e = torch.rad2deg(torch.arccos((n_e * n_ref).sum(-1).clamp(-1, 1)))[msk]
        linhas.append({"nome": c.nome, "origem": origem, "elev": e, "grupo": g, "iou": iou,
                       "normal_erro_medio": float(ang.mean()), "normal_erro_mediana": float(ang.median()),
                       "normal_frac_15": float((ang < 15).float().mean()),
                       "eixo_erro_medio": float(ang_e.mean()), "eixo_erro_mediana": float(ang_e.median())})
        for rotulo, v in meta.items():
            if v["nome"] == c.nome and v["origem"].startswith(origem):
                for nome_img, img in (("normal", n), ("normal_ref", n_ref), ("normal_eixo", n_e)):
                    vis = ((img * 0.5 + 0.5).clamp(0, 1) * 255).byte().cpu().numpy()[..., ::-1].copy()
                    fundo = (~(obj_ref if nome_img == "normal_ref" else obj)).cpu().numpy()
                    vis[fundo] = 255
                    cv2.imwrite(str(pasta / "figs" / f"{rotulo}_{nome_img}.png"), vis)
    resumo = {}
    for g in ("faixa", "acima", "cima"):
        sel = [ln for ln in linhas if ln["grupo"] == g]
        resumo[g] = {"n": len(sel), **{k: float(np.mean([ln[k] for ln in sel]))
                                       for k in ("iou", "normal_erro_medio", "normal_erro_mediana", "normal_frac_15",
                                                 "eixo_erro_medio", "eixo_erro_mediana")}}
    print(a.cena, rot, json.dumps({g: {k: round(v, 3) for k, v in d.items()} for g, d in resumo.items()}), flush=True)
    (pasta / "geometria.json").write_text(json.dumps({"resumo": resumo, "por_vista": linhas}, indent=1))


if __name__ == "__main__":
    main()
