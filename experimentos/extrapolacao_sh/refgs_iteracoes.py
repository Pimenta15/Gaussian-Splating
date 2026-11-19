"""Quanto o número de iterações explica a qualidade do Ref-Gaussian treinado com todas as alturas?

Avalia os checkpoints de 30, 40 e 50 mil iterações (cada um com a malha da mesma iteração) nas 200 vistas de teste
de cada cena, separadas em dentro (< 25°) e acima (>= 25°) da faixa. Com 30 mil, o modelo completo tem o mesmo número
de iterações do 3DGS e quase o mesmo do Ref-Gaussian treinado só na faixa (28 mil).

Roda no contêiner com ref-gaussian/output montado em /opt/ref-gaussian/output e ref_nerf em
/opt/ref-gaussian/data/ref_nerf:
    python /exp/refgs_iteracoes.py
"""
import json
import math
import os
import sys

REFGS = "/opt/ref-gaussian"
sys.path.insert(0, REFGS)
os.chdir(REFGS)

import torch  # noqa: E402
from argparse import ArgumentParser  # noqa: E402
from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import GaussianModel, render_surfel  # noqa: E402
from scene import Scene  # noqa: E402

CENAS = ["ball", "car", "coffee", "helmet", "teapot", "toaster"]
ITERACOES = [30000, 40000, 50000]


def elevacao(fr) -> float:
    t = [r[3] for r in fr["transform_matrix"][:3]]
    return math.degrees(math.asin(t[2] / math.sqrt(sum(x * x for x in t))))


def main():
    saida = "/exp/saidas/refgs_iteracoes.json"
    res = json.load(open(saida)) if os.path.exists(saida) else {}
    for cena in CENAS:
        frames = json.load(open(f"data/ref_nerf/{cena}/transforms_test.json"))["frames"]
        elev = {os.path.basename(fr["file_path"]): elevacao(fr) for fr in frames}
        for it in ITERACOES:
            chave = f"{cena}_{it}"
            if chave in res or not os.path.exists(f"output/{cena}/point_cloud/iteration_{it}"):
                continue
            sys.argv = [sys.argv[0], "-m", f"output/{cena}", "-s", f"data/ref_nerf/{cena}"]
            ap = ArgumentParser()
            mp, op, pp = ModelParams(ap, sentinel=True), OptimizationParams(ap), PipelineParams(ap)
            ap.add_argument("--iteration", default=-1, type=int)
            args = get_combined_args(ap)
            args.data_device = "cpu"
            ds, pipe = mp.extract(args), pp.extract(args)
            with torch.no_grad():
                g = GaussianModel(ds.sh_degree)
                sc = Scene(ds, g, load_iteration=it, shuffle=False)
                op.indirect = 1
                g.load_mesh_from_ply(ds.model_path, it)
                g.env_map.build_mips()
                fundo = torch.tensor([1, 1, 1] if ds.white_background else [0, 0, 0], dtype=torch.float32,
                                     device="cuda")
                grupos = {"faixa": [], "acima": []}
                for v in sc.getTestCameras():
                    v.refl_mask = None
                    img = torch.clamp(render_surfel(v, g, pipe, fundo, srgb=op.srgb, opt=op)["render"], 0, 1)
                    gt = torch.clamp(v.original_image.cuda(), 0, 1)[:3]
                    p = float(-10 * torch.log10(((img - gt) ** 2).mean()).item())
                    grupos["faixa" if elev[v.image_name] < 25 else "acima"].append(p)
            res[chave] = {g_: sum(x) / len(x) for g_, x in grupos.items()}
            res[chave]["todas"] = sum(grupos["faixa"] + grupos["acima"]) / (len(grupos["faixa"]) + len(grupos["acima"]))
            print(chave, json.dumps({k: round(v, 2) for k, v in res[chave].items()}), flush=True)
            json.dump(res, open(saida, "w"), indent=1)
            del g, sc
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
