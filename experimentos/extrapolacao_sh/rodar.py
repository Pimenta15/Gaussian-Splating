"""Cor dependente de vista fora da faixa de elevação filmada: experimento controlado (Shiny Blender).

Um vídeo walkaround filma o objeto sempre mais ou menos na mesma altura. O 3DGS ajusta os
harmônicos esféricos (SH) de cada gaussiana só nas direções observadas; fora delas a cor é
extrapolação pura. Em dados reais não há imagem de referência de cima para medir o efeito;
em dados sintéticos há. Este script:

1. treina 3DGS (gsplat) só com as vistas de treino abaixo de `--banda` graus de elevação;
2. avalia em todas as vistas inéditas (as 200 de teste e as de treino não usadas), anotando a
   elevação de cada uma, para medir a qualidade em função da altura do olhar;
3. compara, sobre o MESMO modelo treinado, correções dos SH sem treinamento:
   - original: SH como saíram do treino;
   - dc: só o termo constante (descarta toda a dependência de vista);
   - global: projeção de norma mínima nas direções da faixa filmada (uma matriz 16x16 para a cena);
   - visib: projeção por gaussiana, com as direções em que ela foi de fato vista nas câmeras de treino,
     ponderadas pela contribuição dela aos pixels (já considera oclusão e enquadramento).

Uso (dentro do container do worker, com vitrine_worker no PYTHONPATH):
    python rodar.py --dados /data --cena car --banda 25 --saida /exp/saidas
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from vitrine_worker.datasets import Camera, foco_das_cameras, ler_transforms, raio_da_cena
from vitrine_worker.metrics import LPIPS, psnr, ssim
from vitrine_worker.sh_observado import direcoes_observadas, matriz_projecao
from vitrine_worker.train_3dgs import Config3DGS, _iniciar, _renderizar

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def base_sh(d: torch.Tensor, grau: int) -> torch.Tensor:
    """Mesma base do gsplat/3DGS (e de sh_observado.base_sh), em torch. d [N,3] -> [N,(grau+1)^2]."""
    x, y, z = d.unbind(1)
    cols = [torch.full_like(x, C0)]
    if grau >= 1:
        cols += [-C1 * y, C1 * z, -C1 * x]
    if grau >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        cols += [C2[0] * xy, C2[1] * yz, C2[2] * (2 * zz - xx - yy), C2[3] * xz, C2[4] * (xx - yy)]
    if grau >= 3:
        cols += [C3[0] * y * (3 * xx - yy), C3[1] * xy * z, C3[2] * y * (4 * zz - xx - yy),
                 C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
                 C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)]
    return torch.stack(cols, 1)


def elevacao(c: Camera) -> float:
    p = c.c2w[:3, 3]
    return math.degrees(math.asin(p[2] / np.linalg.norm(p)))


def azimute(c: Camera) -> float:
    p = c.c2w[:3, 3]
    return math.degrees(math.atan2(p[1], p[0]))


# --------------------------------------------------------------------------------------------------
# treino (mesmo laço de vitrine_worker.train_3dgs.treinar, devolvendo os parâmetros)
# --------------------------------------------------------------------------------------------------
def treinar(cams: list[Camera], iteracoes: int, sh_grau: int, semente: int = 0) -> tuple[dict, float]:
    from gsplat.strategy import DefaultStrategy
    from gsplat.strategy.ops import reset_opa

    cfg = Config3DGS(iteracoes=iteracoes, sh_grau=sh_grau, semente=semente)
    device = torch.device("cuda")
    torch.manual_seed(semente)
    raio = raio_da_cena(cams)
    foco, dist = foco_das_cameras(cams)
    params = _iniciar(cfg, foco, dist, device)
    lrs = {"means": 1.6e-4 * raio, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3,
           "shN": 2.5e-3 / 20}
    otim = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()}
    agenda = torch.optim.lr_scheduler.ExponentialLR(otim["means"], gamma=0.01 ** (1.0 / iteracoes))
    fim_refino = iteracoes // 2
    estrategia = DefaultStrategy(refine_stop_iter=fim_refino, refine_scale2d_stop_iter=fim_refino, verbose=False)
    estrategia.check_sanity(params, otim)
    estado = estrategia.initialize_state(scene_scale=raio)
    fundo = torch.ones(1, 3, device=device)
    imgs = [torch.from_numpy(c.imagem).to(device) for c in cams]
    Ks = [torch.from_numpy(c.K).float().to(device) for c in cams]
    vms = [torch.from_numpy(c.viewmat).float().to(device) for c in cams]
    inicio = time.time()
    for passo in range(iteracoes):
        i = int(torch.randint(len(cams), (1,)))
        grau = min(passo // 1000, sh_grau)
        rgb, _, info = _renderizar(params, Ks[i][None], vms[i][None], cams[i].largura, cams[i].altura, grau, fundo)
        estrategia.step_pre_backward(params, otim, estado, passo, info)
        pred, ref = rgb[0], imgs[i]
        s = ssim(pred.permute(2, 0, 1)[None], ref.permute(2, 0, 1)[None])
        perda = 0.8 * F.l1_loss(pred, ref) + 0.2 * (1 - s)
        perda.backward()
        for o in otim.values():
            o.step()
            o.zero_grad(set_to_none=True)
        agenda.step()
        estrategia.step_post_backward(params, otim, estado, passo, info, packed=False)
        if 0 < passo < fim_refino and passo % estrategia.reset_every == 0:  # bug do reset no gsplat 1.5.3
            reset_opa(params=params, optimizers=otim, state=estado, value=estrategia.prune_opa * 2.0)
        if passo % 2000 == 0 or passo == iteracoes - 1:
            print(f"  passo {passo:6d}/{iteracoes}  perda {float(perda):.4f}  "
                  f"gaussianas {params['means'].shape[0]}  {time.time() - inicio:.0f}s", flush=True)
    return {k: v.detach() for k, v in params.items()}, time.time() - inicio


# --------------------------------------------------------------------------------------------------
# correções dos SH sem treinamento
# --------------------------------------------------------------------------------------------------
def coef(params) -> torch.Tensor:
    return torch.cat([params["sh0"], params["shN"]], 1)  # [N,K,3]


def com_coef(params, c: torch.Tensor) -> dict:
    p = dict(params)
    p["sh0"], p["shN"] = c[:, :1].contiguous(), c[:, 1:].contiguous()
    return p


def variante_dc(params) -> dict:
    c = coef(params).clone()
    c[:, 1:] = 0
    return com_coef(params, c)


def variante_global(params, cams: list[Camera], margem: float = 5.0, lam: float = 1e-2) -> dict:
    el = [elevacao(c) for c in cams]
    grau = int(round(math.sqrt(params["shN"].shape[1] + 1))) - 1
    M = torch.from_numpy(matriz_projecao(direcoes_observadas(min(el), max(el), margem), grau, lam)).float().cuda()
    return com_coef(params, torch.einsum("kj,njc->nkc", M, coef(params)))


@torch.no_grad()
def _gram_por_gaussiana(params, cams: list[Camera], grau: int) -> tuple[torch.Tensor, torch.Tensor]:
    """G_i = média ponderada de B(d_ij)^T B(d_ij) nas câmeras j, com peso = soma dos pesos de mistura
    da gaussiana i nos pixels da vista j (alfa_i * transmitância), obtida pelo gradiente de uma
    renderização com cor constante 1. Considera oclusão e enquadramento de graça."""
    from gsplat import rasterization

    means = params["means"]
    n, k = means.shape[0], (grau + 1) ** 2
    G = torch.zeros(n, k, k, device=means.device)
    W = torch.zeros(n, device=means.device)
    escalas, opac = torch.exp(params["scales"]), torch.sigmoid(params["opacities"])
    for c in cams:
        uns = torch.ones(n, 3, device=means.device, requires_grad=True)
        K = torch.from_numpy(c.K).float().cuda()[None]
        vm = torch.from_numpy(c.viewmat).float().cuda()[None]
        with torch.enable_grad():
            out, _, _ = rasterization(means=means, quats=params["quats"], scales=escalas, opacities=opac,
                                      colors=uns, viewmats=vm, Ks=K, width=c.largura, height=c.altura,
                                      sh_degree=None, packed=False, backgrounds=torch.zeros(1, 3, device="cuda"))
            out[..., 0].sum().backward()
        w = uns.grad[:, 0]
        centro = torch.from_numpy(c.c2w[:3, 3]).float().cuda()
        B = base_sh(F.normalize(means - centro, dim=1), grau)  # direção câmera -> gaussiana, como o gsplat
        for a in range(0, n, 262144):  # em blocos: [n,16,16] temporário cabe na GPU de 6 GB
            b = slice(a, a + 262144)
            G[b] += w[b, None, None] * B[b, :, None] * B[b, None, :]
        W += w
    return G, W


@torch.no_grad()
def variante_visib(params, G: torch.Tensor, W: torch.Tensor, lam: float = 1e-2) -> dict:
    n, k = G.shape[:2]
    c = coef(params)
    L = torch.eye(k, device=G.device) * lam
    L[0, 0] = 0.0
    vistas = W > 1e-6
    novo = c.clone()
    novo[~vistas, 1:] = 0  # nunca vista em nenhuma câmera de treino: fica só a cor média
    idx = vistas.nonzero()[:, 0]
    for a in range(0, len(idx), 131072):
        ii = idx[a:a + 131072]
        Gn = G[ii] / W[ii, None, None]
        M = torch.linalg.solve(Gn + L, Gn)  # c' = (G + L)^-1 G c
        novo[ii] = torch.einsum("nkj,njc->nkc", M, c[ii])
    return com_coef(params, novo)


# --------------------------------------------------------------------------------------------------
# avaliação
# --------------------------------------------------------------------------------------------------
@torch.no_grad()
def renderizar(params, c: Camera) -> torch.Tensor:
    grau = int(round(math.sqrt(params["shN"].shape[1] + 1))) - 1
    K = torch.from_numpy(c.K).float().cuda()[None]
    vm = torch.from_numpy(c.viewmat).float().cuda()[None]
    rgb, _, _ = _renderizar(params, K, vm, c.largura, c.altura, grau, torch.ones(1, 3, device="cuda"))
    return rgb[0].clamp(0, 1)


@torch.no_grad()
def avaliar(variantes: dict[str, dict], vistas: list[tuple[str, Camera]], lp: LPIPS) -> list[dict]:
    linhas = []
    for origem, c in vistas:
        ref = torch.from_numpy(c.imagem).cuda()
        linha = {"nome": c.nome, "origem": origem, "elev": elevacao(c), "azim": azimute(c)}
        for nome, p in variantes.items():
            pred = renderizar(p, c)
            a, b = pred.permute(2, 0, 1)[None], ref.permute(2, 0, 1)[None]
            linha[nome] = {"psnr": psnr(pred, ref), "ssim": float(ssim(a, b)), "lpips": lp(a, b)}
        linhas.append(linha)
    return linhas


def salvar_figuras(variantes: dict[str, dict], vistas: list[tuple[str, Camera]], pasta: Path) -> None:
    """Para algumas vistas (a mais alta, uma intermediária e uma na faixa), salva GT e cada variante."""
    pasta.mkdir(parents=True, exist_ok=True)
    por_elev = sorted(vistas, key=lambda v: elevacao(v[1]))
    escolha = {"cima": por_elev[-1], "meio": min(por_elev, key=lambda v: abs(elevacao(v[1]) - 40)),
               "faixa": por_elev[len(por_elev) // 8]}
    meta = {}
    for rotulo, (origem, c) in escolha.items():
        cv2.imwrite(str(pasta / f"{rotulo}_gt.png"), (c.imagem[..., ::-1] * 255).astype(np.uint8))
        for nome, p in variantes.items():
            img = renderizar(p, c).cpu().numpy()
            cv2.imwrite(str(pasta / f"{rotulo}_{nome}.png"), (img[..., ::-1] * 255).astype(np.uint8))
        meta[rotulo] = {"nome": c.nome, "origem": origem, "elev": elevacao(c), "azim": azimute(c)}
    (pasta / "vistas.json").write_text(json.dumps(meta, indent=1))


def resumo(linhas: list[dict], variantes: list[str], banda: float) -> dict:
    grupos = {"faixa": lambda e: e < banda, "acima": lambda e: banda <= e < 50, "cima": lambda e: e >= 50}
    out = {}
    for g, f in grupos.items():
        sel = [ln for ln in linhas if f(ln["elev"])]
        out[g] = {"n": len(sel)}
        for v in variantes:
            if sel:
                out[g][v] = {m: float(np.mean([ln[v][m] for ln in sel])) for m in ("psnr", "ssim", "lpips")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="/data")
    ap.add_argument("--cena", required=True)
    ap.add_argument("--banda", type=float, default=25.0, help="elevação máxima (graus) das vistas de treino")
    ap.add_argument("--iteracoes", type=int, default=30000)
    ap.add_argument("--sh-grau", type=int, default=3)
    ap.add_argument("--saida", default="/exp/saidas")
    a = ap.parse_args()

    rotulo = f"b{a.banda:g}_sh{a.sh_grau}" if a.banda < 90 else f"completo_sh{a.sh_grau}"
    saida = Path(a.saida) / a.cena / rotulo
    saida.mkdir(parents=True, exist_ok=True)
    pasta = Path(a.dados) / a.cena
    treino_todo = ler_transforms(pasta / "transforms_train.json", fundo_branco=True)
    teste = ler_transforms(pasta / "transforms_test.json", fundo_branco=True)
    treino = [c for c in treino_todo if 0 <= elevacao(c) < a.banda]
    nao_usadas = [c for c in treino_todo if elevacao(c) >= a.banda]
    print(f"{a.cena} {rotulo}: {len(treino)} vistas de treino "
          f"(elevação {min(map(elevacao, treino)):.1f}..{max(map(elevacao, treino)):.1f}), "
          f"{len(teste)} de teste, {len(nao_usadas)} de treino não usadas", flush=True)

    params, tempo = treinar(treino, a.iteracoes, a.sh_grau)
    torch.save({k: v.cpu() for k, v in params.items()}, saida / "params.pt")

    t0 = time.time()
    variantes = {"original": params}
    tempos_corr = {}
    if a.sh_grau > 0 and a.banda < 90:
        t = time.time()
        variantes["dc"] = variante_dc(params)
        tempos_corr["dc"] = time.time() - t
        t = time.time()
        variantes["global"] = variante_global(params, treino)
        torch.cuda.synchronize()
        tempos_corr["global"] = time.time() - t
        t = time.time()
        G, W = _gram_por_gaussiana(params, treino, a.sh_grau)
        variantes["visib"] = variante_visib(params, G, W, 1e-2)
        torch.cuda.synchronize()
        tempos_corr["visib"] = time.time() - t
        for lam in (1e-3, 1e-1):
            variantes[f"visib_l{lam:g}"] = variante_visib(params, G, W, lam)
        frac_vistas = float((W > 1e-6).float().mean())
        del G
    else:
        frac_vistas = None
    vistas = [("teste", c) for c in teste] + [("treino_nao_usada", c) for c in nao_usadas]
    linhas = avaliar(variantes, vistas, LPIPS("cuda"))
    salvar_figuras(variantes, vistas, saida / "figs")
    res = {
        "cena": a.cena, "banda": a.banda, "sh_grau": a.sh_grau, "iteracoes": a.iteracoes,
        "n_treino": len(treino), "elev_treino": [elevacao(c) for c in treino],
        "n_gaussianas": int(params["means"].shape[0]), "tempo_treino_s": tempo,
        "tempo_correcao_s": tempos_corr, "tempo_avaliacao_s": time.time() - t0,
        "frac_gaussianas_vistas": frac_vistas,
        "resumo": resumo(linhas, list(variantes), a.banda), "por_vista": linhas,
    }
    (saida / "resultado.json").write_text(json.dumps(res, indent=1))
    for g, r in res["resumo"].items():
        txt = "  ".join(f"{v} {r[v]['psnr']:.2f}" for v in variantes if v in r)
        print(f"  {g:6s} (n={r['n']:3d}): {txt}", flush=True)
    print("ok", flush=True)


if __name__ == "__main__":
    main()
