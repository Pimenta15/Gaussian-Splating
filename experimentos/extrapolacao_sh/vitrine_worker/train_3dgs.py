"""Rota rápida: 3D Gaussian Splatting com o gsplat (Apache-2.0).

Treina a partir de câmeras com pose conhecida (`datasets.Camera`), avalia nas vistas de validação
e exporta a cena em .ply (formato padrão do 3DGS) e .splat (visualizador web).
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .datasets import Camera, foco_das_cameras, raio_da_cena
from .export import para_splat
from .metrics import LPIPS, psnr, ssim


@dataclass
class Config3DGS:
    iteracoes: int = 7000
    sh_grau: int = 3
    n_pontos_iniciais: int = 100_000
    frac_cubo_inicial: float = 0.3
    lambda_ssim: float = 0.2
    fundo_branco: bool = True
    log_cada: int = 100
    semente: int = 0


def _rgb_para_sh0(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / 0.28209479177387814


def _iniciar(cfg: Config3DGS, foco: np.ndarray, dist: float, device) -> torch.nn.ParameterDict:
    """Pontos aleatórios num cubo em volta do objeto (sem nuvem do COLMAP, como no 3DGS sintético).

    O 3DGS usa um cubo de meia aresta 1,3 com câmeras a ~4 unidades; aqui a mesma proporção em relação
    à distância câmera-objeto. Um cubo maior põe pontos colados nas câmeras, que viram borrões na tela.
    """
    g = torch.Generator(device="cpu").manual_seed(cfg.semente)
    n = cfg.n_pontos_iniciais
    extensao = cfg.frac_cubo_inicial * dist
    means = (torch.rand(n, 3, generator=g) * 2 - 1) * extensao + torch.from_numpy(foco).float()
    # escala inicial pela distância média aos vizinhos (aproximação pelo volume do cubo)
    escala = (2 * extensao) / (n ** (1 / 3))
    scales = torch.full((n, 3), math.log(escala))
    quats = F.normalize(torch.randn(n, 4, generator=g), dim=-1)
    opacities = torch.logit(torch.full((n,), 0.1))
    cores = torch.rand(n, 3, generator=g)
    sh0 = _rgb_para_sh0(cores).unsqueeze(1)
    shN = torch.zeros(n, (cfg.sh_grau + 1) ** 2 - 1, 3)
    params = {"means": means, "scales": scales, "quats": quats, "opacities": opacities, "sh0": sh0, "shN": shN}
    return torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device)) for k, v in params.items()})


def _renderizar(params, cams_k: torch.Tensor, viewmats: torch.Tensor, w: int, h: int, sh_grau: int, fundo,
                absgrad: bool = False):
    from gsplat import rasterization  # noqa: PLC0415 - importa CUDA só no worker

    cores = torch.cat([params["sh0"], params["shN"]], dim=1)
    rgb, alfa, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=cores,
        viewmats=viewmats,
        Ks=cams_k,
        width=w,
        height=h,
        sh_degree=sh_grau,
        backgrounds=fundo,
        # no modo packed (default) o gsplat 1.5.3 valida `backgrounds` com a dimensão errada;
        # unpacked também é o que a DefaultStrategy espera em step_post_backward(packed=False)
        packed=False,
        absgrad=absgrad,
    )
    return rgb, alfa, info


def treinar(
    treino: list[Camera],
    validacao: list[Camera],
    saida: Path,
    cfg: Config3DGS | None = None,
    ao_logar: Callable[[int, dict], None] | None = None,
) -> dict:
    """Treina, avalia e exporta. Devolve as métricas de validação e os caminhos gerados."""
    from gsplat.strategy import DefaultStrategy  # noqa: PLC0415
    from gsplat.strategy.ops import reset_opa  # noqa: PLC0415

    cfg = cfg or Config3DGS()
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(cfg.semente)

    raio = raio_da_cena(treino)
    foco, dist = foco_das_cameras(treino)
    params = _iniciar(cfg, foco, dist, device)
    lrs = {"means": 1.6e-4 * raio, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    otimizadores = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()}
    agendador = torch.optim.lr_scheduler.ExponentialLR(otimizadores["means"], gamma=0.01 ** (1.0 / cfg.iteracoes))

    fim_refino = int(cfg.iteracoes * 0.5)
    # refine_scale2d_stop_iter > 0 liga a poda de gaussianas grandes na tela, como no 3DGS original
    # (no gsplat ela vem desligada por padrão)
    estrategia = DefaultStrategy(refine_stop_iter=fim_refino, refine_scale2d_stop_iter=fim_refino, verbose=False)
    estrategia.check_sanity(params, otimizadores)
    estado = estrategia.initialize_state(scene_scale=raio)

    fundo = torch.ones(1, 3, device=device) if cfg.fundo_branco else torch.zeros(1, 3, device=device)
    imgs = [torch.from_numpy(c.imagem).to(device) for c in treino]
    Ks = [torch.from_numpy(c.K).float().to(device) for c in treino]
    vms = [torch.from_numpy(c.viewmat).float().to(device) for c in treino]

    inicio = time.time()
    for passo in range(cfg.iteracoes):
        i = int(torch.randint(len(treino), (1,)))
        grau = min(passo // 1000, cfg.sh_grau)  # aumenta o grau dos harmônicos aos poucos
        rgb, _, info = _renderizar(params, Ks[i][None], vms[i][None], treino[i].largura, treino[i].altura, grau, fundo)
        estrategia.step_pre_backward(params, otimizadores, estado, passo, info)
        pred, ref = rgb[0], imgs[i]
        l1 = F.l1_loss(pred, ref)
        s = ssim(pred.permute(2, 0, 1)[None], ref.permute(2, 0, 1)[None])
        perda = (1 - cfg.lambda_ssim) * l1 + cfg.lambda_ssim * (1 - s)
        perda.backward()
        for opt in otimizadores.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        agendador.step()
        estrategia.step_post_backward(params, otimizadores, estado, passo, info, packed=False)
        # No gsplat 1.5.3 a condição do reset de opacidade é `step % reset_every == 0 & step > 0`: pela
        # precedência do Python vira `... == (0 & step) > 0`, sempre falsa, e o reset nunca acontece.
        # Sem ele os floaters (gaussianas soltas perto das câmeras) não são eliminados; refazemos aqui.
        # Corrigido no main do gsplat; a 1.5.3 é a última com wheel para PyTorch 2.0 + CUDA 11.8.
        if 0 < passo < fim_refino and passo % estrategia.reset_every == 0:
            reset_opa(params=params, optimizers=otimizadores, state=estado, value=estrategia.prune_opa * 2.0)
        if ao_logar and (passo % cfg.log_cada == 0 or passo == cfg.iteracoes - 1):
            ao_logar(passo, {"perda": float(perda), "n_gaussianas": int(params["means"].shape[0])})

    tempo_treino = time.time() - inicio
    metricas = avaliar(params, validacao, cfg, fundo)
    metricas.update({"tempo_treino_s": tempo_treino, "n_gaussianas": int(params["means"].shape[0])})

    caminhos = exportar(params, saida)
    return {"metricas": metricas, "arquivos": caminhos}


@torch.no_grad()
def avaliar(params, cams: list[Camera], cfg: Config3DGS, fundo) -> dict:
    device = fundo.device
    lp = LPIPS(str(device))
    ps, ss, ls, tempos = [], [], [], []
    for c in cams:
        K = torch.from_numpy(c.K).float().to(device)[None]
        vm = torch.from_numpy(c.viewmat).float().to(device)[None]
        torch.cuda.synchronize()
        t = time.time()
        rgb, _, _ = _renderizar(params, K, vm, c.largura, c.altura, cfg.sh_grau, fundo)
        torch.cuda.synchronize()
        tempos.append(time.time() - t)
        pred = rgb[0].clamp(0, 1).permute(2, 0, 1)[None]
        ref = torch.from_numpy(c.imagem).to(device).permute(2, 0, 1)[None]
        ps.append(psnr(pred, ref))
        ss.append(float(ssim(pred, ref)))
        ls.append(lp(pred, ref))
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "lpips": float(np.mean(ls)),
            "fps": float(1.0 / np.mean(tempos)), "psnr_min": float(np.min(ps)), "n_vistas_validacao": len(cams)}


def exportar(params, saida: Path) -> dict:
    p = {k: v.detach().cpu().numpy() for k, v in params.items()}
    splat = saida / "cena.splat"
    splat.write_bytes(para_splat(p["means"], p["scales"], p["quats"], p["opacities"], p["sh0"]))
    ply = saida / "cena.ply"
    _salvar_ply(p, ply)
    return {"splat": str(splat), "ply": str(ply)}


def _salvar_ply(p: dict, caminho: Path) -> None:
    """PLY no layout do 3DGS original (lido por SuperSplat, gsplat.js e afins)."""
    n = p["means"].shape[0]
    f_dc = p["sh0"].reshape(n, 3)
    f_rest = p["shN"].transpose(0, 2, 1).reshape(n, -1)  # [N, 3*K] canal a canal, como no 3DGS
    campos = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
              + [f"f_rest_{i}" for i in range(f_rest.shape[1])] + ["opacity"]
              + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    dados = np.concatenate([p["means"], np.zeros((n, 3)), f_dc, f_rest, p["opacities"].reshape(n, 1),
                            p["scales"], p["quats"]], axis=1).astype("<f4")
    cabecalho = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    cabecalho += "".join(f"property float {c}\n" for c in campos) + "end_header\n"
    with open(caminho, "wb") as f:
        f.write(cabecalho.encode())
        f.write(dados.tobytes())
