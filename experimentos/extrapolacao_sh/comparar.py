"""Compara correções leves, aplicadas a um modelo 3DGS já treinado, por qualidade, tempo e armazenamento.

Variantes (todas sem novo treino, sobre o params.pt salvo por rodar.py):
  original     3DGS como saiu do treino (SH de grau 3, 48 números de cor por Gaussiana)
  grau0/1/2    truncamento dos SH no grau indicado, como fazem visualizadores leves e formatos compactos
  adaptativo   grau escolhido por Gaussiana: o menor grau cuja cor, nas direções em que ela foi vista,
               difere menos de TAU da original (critério no espírito de Papantonakis et al., 2024)
  global       projeção com a matriz de Gram global da faixa (rodar.variante_global)
  visib        projeção com a matriz de Gram por Gaussiana (rodar.variante_visib)
  visib_adapt  projeção por visibilidade seguida do grau adaptativo: a projeção zera o que nenhuma câmera
               viu, e o grau adaptativo descarta esses coeficientes do armazenamento

Também salva, para a demonstração interativa, as imagens de uma varredura de elevação (uma vista por faixa de
5 graus), com referência, variantes, modelo treinado com todas as alturas e Ref-Gaussian quando existirem.

    python comparar.py --cena car --banda 25
"""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from rodar import (_gram_por_gaussiana, avaliar, coef, com_coef, elevacao, azimute, renderizar, variante_dc,
                   variante_global, variante_visib)
from vitrine_worker.datasets import ler_transforms
from vitrine_worker.metrics import LPIPS
from vitrine_worker.sh_observado import base_sh as base_sh_np, direcoes_observadas

TAU = 0.01  # diferença de cor tolerada (escala 0..1) nas direções observadas
GEOM = 3 + 3 + 4 + 1  # posição, escala, rotação, opacidade


def truncar(params, grau: int) -> dict:
    c = coef(params).clone()
    c[:, (grau + 1) ** 2:] = 0
    return com_coef(params, c)


@torch.no_grad()
def grau_adaptativo(params, G: torch.Tensor, W: torch.Tensor, tau: float = TAU) -> tuple[dict, torch.Tensor]:
    """Menor grau por Gaussiana cuja cor nas direções vistas (média ponderada pela visibilidade) muda < tau."""
    c = coef(params)
    n, k = c.shape[:2]
    grau = torch.zeros(n, dtype=torch.long, device=c.device)
    vistas = (W > 1e-6).nonzero()[:, 0]
    for a in range(0, len(vistas), 131072):
        ii = vistas[a:a + 131072]
        # G por Gaussiana [n,k,k] (visibilidade) ou uma só [k,k] para a cena (global, sem imagens)
        Gn = G[ii] / W[ii, None, None] if G.dim() == 3 else G.expand(len(ii), k, k)
        escolhido = torch.full((len(ii),), 3, dtype=torch.long, device=c.device)
        for l in (2, 1, 0):
            d = c[ii].clone()
            d[:, :(l + 1) ** 2] = 0  # o que o truncamento no grau l descarta
            err = torch.einsum("nkc,nkj,njc->n", d, Gn, d) / 3
            escolhido = torch.where(err.clamp_min(0).sqrt() <= tau, torch.full_like(escolhido, l), escolhido)
        grau[ii] = escolhido
    novo = c.clone()
    keep = (grau[:, None] + 1) ** 2  # coeficientes mantidos por Gaussiana
    novo[torch.arange(k, device=c.device)[None, :] >= keep] = 0
    return com_coef(params, novo), grau


def bytes_por_gaussiana(graus: torch.Tensor | int) -> float:
    k = (graus + 1) ** 2 if isinstance(graus, int) else ((graus + 1) ** 2).float().mean().item()
    return 4.0 * (GEOM + 3 * k)


def varredura(teste, nao_usadas, passo: float = 5.0) -> list[tuple[str, object]]:
    """Uma vista por faixa de elevação, com o azimute mais próximo do da primeira vista escolhida."""
    todas = [("teste", c) for c in teste] + [("treino_nao_usada", c) for c in nao_usadas]
    ref_az = None
    out = []
    for lo in np.arange(0, 90, passo):
        faixa = [v for v in todas if lo <= elevacao(v[1]) < lo + passo]
        if not faixa:
            continue
        if ref_az is None:
            ref_az = azimute(faixa[0][1])
        out.append(min(faixa, key=lambda v: abs((azimute(v[1]) - ref_az + 180) % 360 - 180)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="/data")
    ap.add_argument("--cena", required=True)
    ap.add_argument("--banda", type=float, default=25.0)
    ap.add_argument("--saida", default="/exp/saidas")
    a = ap.parse_args()
    pasta = Path(a.saida) / a.cena / f"b{a.banda:g}_sh3"
    params = {k: v.cuda() for k, v in torch.load(pasta / "params.pt").items()}
    treino_todo = ler_transforms(Path(a.dados) / a.cena / "transforms_train.json", fundo_branco=True)
    teste = ler_transforms(Path(a.dados) / a.cena / "transforms_test.json", fundo_branco=True)
    treino = [c for c in treino_todo if 0 <= elevacao(c) < a.banda]
    nao_usadas = [c for c in treino_todo if elevacao(c) >= a.banda]

    tempos, var = {}, {"original": params}
    for g in (0, 1, 2):
        torch.cuda.synchronize(); t = time.time()
        var[f"grau{g}"] = truncar(params, g) if g else variante_dc(params)
        torch.cuda.synchronize(); tempos[f"grau{g}"] = time.time() - t
    torch.cuda.synchronize(); t = time.time()
    var["global"] = variante_global(params, treino)
    torch.cuda.synchronize(); tempos["global"] = time.time() - t
    t = time.time()
    G, W = _gram_por_gaussiana(params, treino, 3)
    torch.cuda.synchronize(); tempos["gram"] = time.time() - t
    t = time.time()
    var["visib"] = variante_visib(params, G, W, 1e-2)
    torch.cuda.synchronize(); tempos["visib"] = tempos["gram"] + time.time() - t
    t = time.time()
    var["adaptativo"], g_ad = grau_adaptativo(params, G, W)
    torch.cuda.synchronize(); tempos["adaptativo"] = tempos["gram"] + time.time() - t
    t = time.time()
    var["visib_adapt"], g_va = grau_adaptativo(var["visib"], G, W)
    torch.cuda.synchronize(); tempos["visib_adapt"] = tempos["visib"] + time.time() - t
    del G
    # versão leve: projeção global + grau adaptativo com a matriz de Gram global (sem imagens nem GPU)
    t = time.time()
    el = [elevacao(c) for c in treino]
    Bg = torch.from_numpy(base_sh_np(direcoes_observadas(min(el), max(el), 5.0), 3)).float().cuda()
    Gg = Bg.T @ Bg / len(Bg)
    var["global_adapt"], g_ga = grau_adaptativo(var["global"], Gg, torch.ones(len(W), device=W.device))
    torch.cuda.synchronize(); tempos["global_adapt"] = tempos["global"] + time.time() - t
    n = int(params["means"].shape[0])
    armaz = {"original": bytes_por_gaussiana(3), "grau0": bytes_por_gaussiana(0), "grau1": bytes_por_gaussiana(1),
             "grau2": bytes_por_gaussiana(2), "global": bytes_por_gaussiana(3), "visib": bytes_por_gaussiana(3),
             "adaptativo": bytes_por_gaussiana(g_ad), "visib_adapt": bytes_por_gaussiana(g_va),
             "global_adapt": bytes_por_gaussiana(g_ga)}
    hist = {k: [int((g == l).sum()) for l in range(4)]
            for k, g in (("adaptativo", g_ad), ("visib_adapt", g_va), ("global_adapt", g_ga))}

    vistas = [("teste", c) for c in teste] + [("treino_nao_usada", c) for c in nao_usadas]
    linhas = avaliar(var, vistas, LPIPS("cuda"))
    res = {"cena": a.cena, "banda": a.banda, "n_gaussianas": n, "tau": TAU, "tempo_s": tempos,
           "bytes_por_gaussiana": armaz, "mb": {k: v * n / 2 ** 20 for k, v in armaz.items()},
           "hist_grau": hist, "por_vista": linhas}
    (pasta / "comparacao.json").write_text(json.dumps(res, indent=1))

    # imagens da varredura para a demonstração interativa
    dest = pasta / "explorar"
    dest.mkdir(exist_ok=True)
    extra = {}
    f_c = Path(a.saida) / a.cena / "completo_sh3" / "params.pt"
    if f_c.exists():
        extra["completo"] = {k: v.cuda() for k, v in torch.load(f_c).items()}
    meta_rg = Path(a.saida).parent / "dados_banda" / f"{a.cena}_b{a.banda:g}" / "transforms_test.json"
    rg_dir = Path(a.saida) / a.cena / f"refgs_b{a.banda:g}" / "test" / "renders" / "rgb"
    idx_rg = {}
    if meta_rg.exists() and rg_dir.exists():
        for i, fr in enumerate(json.loads(meta_rg.read_text())["frames"]):
            idx_rg[(fr["origem"], Path(fr["file_path"]).name)] = i
    meta = []
    for j, (origem, c) in enumerate(varredura(teste, nao_usadas)):
        def salvar(nome, img):
            img = cv2.resize((img * 255).clip(0, 255).astype(np.uint8)[..., ::-1], (400, 400),
                             interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dest / f"{j:02d}_{nome}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        salvar("gt", c.imagem)
        for nome in ("original", "grau0", "grau1", "global", "global_adapt", "visib", "visib_adapt"):
            salvar(nome, renderizar(var[nome], c).cpu().numpy())
        for nome, p in extra.items():
            salvar(nome, renderizar(p, c).cpu().numpy())
        k = idx_rg.get((origem, c.nome))
        if k is not None:
            rg = cv2.imread(str(rg_dir / f"{k:05d}.png"))[..., ::-1].astype(np.float32) / 255
            salvar("refgs", rg)
        lin = next(ln for ln in linhas if ln["nome"] == c.nome and ln["origem"] == origem)
        meta.append({"i": j, "elev": elevacao(c), "azim": azimute(c), "origem": origem,
                     "psnr": {v: lin[v]["psnr"] for v in var}, "tem_refgs": k is not None,
                     "tem_completo": "completo" in extra})
    (dest / "meta.json").write_text(json.dumps(meta, indent=1))
    print(a.cena, json.dumps({"tempo_s": {k: round(v, 3) for k, v in tempos.items()},
                              "mb": {k: round(v, 1) for k, v in res["mb"].items()}}), flush=True)


if __name__ == "__main__":
    main()
