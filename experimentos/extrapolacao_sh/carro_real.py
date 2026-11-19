"""Carro real (Porsche 911, vídeo de toca-discos, câmeras entre 0,4° e 3,8° de elevação).

Compara o modelo com os SH como saíram do treino (cena_sem_projecao.ply) e com a projeção global nas
direções filmadas (cena.ply, o que a Vitrine 3D publica):
  - fidelidade na faixa filmada: PSNR entre as duas renderizações numa volta completa a 2°;
  - efeito fora da faixa: as duas renderizações a 20°, 45°, 70° e 89° (salvas para a figura).

    python carro_real.py /outputs/workspaces/exp_porsche_artigo/modelo /exp/saidas/porsche
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from vitrine_worker.sh_observado import ler_ply
from vitrine_worker.train_carro import renderizar_vista


def carregar(ply: Path) -> dict:
    nomes, d = ler_ply(ply)
    i = {n: k for k, n in enumerate(nomes)}
    rest = sorted((n for n in nomes if n.startswith("f_rest_")), key=lambda n: int(n.split("_")[2]))
    k = len(rest) // 3
    t = lambda cols: torch.from_numpy(np.ascontiguousarray(d[:, [i[c] for c in cols]])).float().cuda()  # noqa: E731
    return {"means": t(["x", "y", "z"]), "scales": t([f"scale_{j}" for j in range(3)]),
            "quats": t([f"rot_{j}" for j in range(4)]), "opacities": t(["opacity"])[:, 0],
            "sh0": t([f"f_dc_{j}" for j in range(3)])[:, None, :],
            "shN": t(rest).view(-1, 3, k).transpose(1, 2).contiguous()}


def pos(az_graus, el_graus, dist=4.6):
    a, e = math.radians(az_graus), math.radians(el_graus)
    return [dist * math.cos(e) * math.cos(a), dist * math.cos(e) * math.sin(a), dist * math.sin(e)]


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) / 255 - b.astype(np.float32) / 255) ** 2)
    return float(10 * np.log10(1 / max(mse, 1e-12)))


def main():
    modelo, saida = Path(sys.argv[1]), Path(sys.argv[2])
    saida.mkdir(parents=True, exist_ok=True)
    antes, depois = carregar(modelo / "cena_sem_projecao.ply"), carregar(modelo / "cena.ply")
    res = {"n_gaussianas": int(antes["means"].shape[0])}
    for el in (2.0, 10.0, 20.0, 45.0, 70.0, 89.0):
        ps = []
        for az in range(0, 360, 10):
            a, b = renderizar_vista(antes, pos(az, el)), renderizar_vista(depois, pos(az, el))
            ps.append(psnr(a, b))
            if az == 310:  # vista 3/4 dianteira
                cv2.imwrite(str(saida / f"el{int(el):02d}_antes.png"), a)
                cv2.imwrite(str(saida / f"el{int(el):02d}_depois.png"), b)
        res[f"el{int(el)}"] = {"psnr_antes_vs_depois_medio": float(np.mean(ps)), "min": float(np.min(ps))}
        print(f"elevação {el:4.0f}°: PSNR(antes, depois) = {np.mean(ps):.2f} dB (mín {np.min(ps):.2f})", flush=True)
    (saida / "resultado.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
