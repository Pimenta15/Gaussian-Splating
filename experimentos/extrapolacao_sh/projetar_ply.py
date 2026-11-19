"""Correção leve, sem treino, de um modelo 3DGS salvo em .ply (formato do 3DGS original).

Aplica a projeção dos harmônicos esféricos no subespaço observado pelas câmeras (Eq. de projeção do artigo,
versão global) e, opcionalmente, descarta os graus que deixam de ser usados. Roda em CPU, só com numpy,
em poucos segundos para centenas de milhares de Gaussianas. O resultado abre em qualquer visualizador de
3DGS que leia .ply com SH (por exemplo, o editor SuperSplat no navegador).

    python projetar_ply.py modelo.ply modelo_proj.ply --elev 0.4 3.8
    python projetar_ply.py modelo.ply modelo_proj.ply --cameras cameras.json   # elevações lidas das câmeras
    python projetar_ply.py modelo.ply modelo_proj.ply --elev 0.4 3.8 --lam 0.01 --relatorio

--elev: elevação mínima e máxima (graus) das câmeras que filmaram o objeto, com o eixo z para cima.
--cameras: cameras.json do 3DGS original (posições das câmeras); o centro é a média das posições.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sh_observado import (base_sh, direcoes_observadas, gravar_ply, ler_ply, matriz_projecao, projetar)


def elevacoes_das_cameras(caminho: Path) -> tuple[float, float]:
    cams = json.loads(caminho.read_text())
    pos = np.array([c["position"] for c in cams], dtype=np.float64)
    rel = pos - pos.mean(0)
    el = np.degrees(np.arcsin(rel[:, 2] / np.linalg.norm(rel, axis=1)))
    return float(el.min()), float(el.max())


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada", type=Path)
    ap.add_argument("saida", type=Path)
    ap.add_argument("--elev", type=float, nargs=2, metavar=("MIN", "MAX"))
    ap.add_argument("--cameras", type=Path)
    ap.add_argument("--lam", type=float, default=1e-2, help="regularização (padrão 1e-2)")
    ap.add_argument("--relatorio", action="store_true", help="imprime o espectro da matriz de Gram")
    a = ap.parse_args()
    if a.elev:
        emin, emax = a.elev
    elif a.cameras:
        emin, emax = elevacoes_das_cameras(a.cameras)
    else:
        ap.error("informe --elev MIN MAX ou --cameras cameras.json")

    t0 = time.time()
    nomes, d = ler_ply(a.entrada)
    idx = {n: i for i, n in enumerate(nomes)}
    rest = sorted((n for n in nomes if n.startswith("f_rest_")), key=lambda n: int(n.split("_")[2]))
    k = len(rest) // 3
    grau = int(round(np.sqrt(k + 1))) - 1
    dirs = direcoes_observadas(emin, emax)
    M = matriz_projecao(dirs, grau, a.lam)
    sh0 = d[:, [idx[f"f_dc_{i}"] for i in range(3)]].reshape(-1, 1, 3)
    shN = d[:, [idx[n] for n in rest]].reshape(-1, 3, k).transpose(0, 2, 1)
    sh0b, shNb = projetar(sh0, shN, M)
    d[:, [idx[f"f_dc_{i}"] for i in range(3)]] = sh0b.reshape(-1, 3)
    d[:, [idx[n] for n in rest]] = shNb.transpose(0, 2, 1).reshape(-1, 3 * k)
    gravar_ply(a.saida, nomes, d)
    dt = time.time() - t0

    print(f"{len(d)} Gaussianas, SH de grau {grau}, câmeras entre {emin:.1f} e {emax:.1f} graus, "
          f"lambda={a.lam:g}: {dt:.1f} s")
    if a.relatorio:
        B = base_sh(direcoes_observadas(emin, emax, margem=0.0), grau)  # a faixa filmada, como na Fig. do espectro
        s = np.linalg.eigvalsh(B.T @ B / len(B))[::-1]
        s = s / s[0]
        print("autovalores da matriz de Gram (normalizados):", " ".join(f"{x:.1e}" for x in s))
        print(f"{int((s < 1e-2).sum())} de {len(s)} combinações de coeficientes abaixo de 1% do maior "
              f"(quase invisíveis nas câmeras)")
        antes = np.concatenate([sh0, shN], 1)
        depois = np.concatenate([sh0b, shNb], 1)
        e_a = np.square(antes[:, 1:]).sum((1, 2)).mean()
        e_d = np.square(depois[:, 1:]).sum((1, 2)).mean()
        print(f"energia média dos termos direcionais: {e_a:.4f} -> {e_d:.4f} ({100 * (1 - e_d / e_a):.0f}% removida)")


if __name__ == "__main__":
    main()
