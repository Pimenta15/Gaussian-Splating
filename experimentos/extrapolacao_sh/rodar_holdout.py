"""3DGS treinado com todas as alturas, sem as fotos de avaliação (divisão de prepara_holdout.py).

Mesmo treino de rodar.py (30 mil iterações, SH de grau 3); avalia nas 200 vistas de teste e nas vistas de treino
separadas, que o modelo nunca viu. Grava saidas/<cena>/holdout_sh3/{params.pt,resultado.json}.

    python rodar_holdout.py --dados /dados/holdout --cena car
"""
import argparse
import json
import time
from pathlib import Path

import torch
from rodar import avaliar, elevacao, treinar
from vitrine_worker.datasets import ler_transforms
from vitrine_worker.metrics import LPIPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="/dados/holdout")
    ap.add_argument("--cena", required=True)
    ap.add_argument("--iteracoes", type=int, default=30000)
    ap.add_argument("--saida", default="/exp/saidas")
    a = ap.parse_args()
    pasta = Path(a.dados) / a.cena
    saida = Path(a.saida) / a.cena / "holdout_sh3"
    saida.mkdir(parents=True, exist_ok=True)
    treino = [c for c in ler_transforms(pasta / "transforms_train.json", fundo_branco=True) if elevacao(c) >= 0]
    teste = ler_transforms(pasta / "transforms_test.json", fundo_branco=True)
    origens = [fr.get("origem", "teste") for fr in json.loads((pasta / "transforms_test.json").read_text())["frames"]]
    print(f"{a.cena}: {len(treino)} vistas de treino (elevação {min(map(elevacao, treino)):.1f}.."
          f"{max(map(elevacao, treino)):.1f}), {origens.count('treino_separada')} separadas", flush=True)
    params, tempo = treinar(treino, a.iteracoes, 3)
    torch.save({k: v.cpu() for k, v in params.items()}, saida / "params.pt")
    t0 = time.time()
    linhas = avaliar({"original": params}, list(zip(origens, teste)), LPIPS("cuda"))
    res = {"cena": a.cena, "iteracoes": a.iteracoes, "n_treino": len(treino),
           "elev_treino": [elevacao(c) for c in treino], "n_gaussianas": int(params["means"].shape[0]),
           "tempo_treino_s": tempo, "tempo_avaliacao_s": time.time() - t0, "por_vista": linhas}
    (saida / "resultado.json").write_text(json.dumps(res, indent=1))
    sep = [ln["original"]["psnr"] for ln in linhas if ln["origem"] == "treino_separada" and ln["elev"] >= 50]
    print(f"{a.cena}: separadas de cima (n={len(sep)}) PSNR {sum(sep) / max(len(sep), 1):.2f}", flush=True)
    print("ok", flush=True)


if __name__ == "__main__":
    main()
