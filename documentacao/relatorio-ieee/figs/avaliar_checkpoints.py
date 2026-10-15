"""Avalia checkpoints intermediarios e o efeito da malha usada na visibilidade.

O eval.py oficial sempre carrega a malha da ultima iteracao (test_<max>.ply). Aqui cada
combinacao (checkpoint de Gaussianas, malha) e avaliada nas mesmas vistas de teste.

Rodar a partir da pasta ref-gaussian, no ambiente do compilador:
  run_ref_gaussian.bat ..\\documentacao\\relatorio-ieee\\figs\\avaliar_checkpoints.py -m output/helmet --white_background
Saida: figs/<cena>_checkpoints.json e uma tabela no terminal.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())  # modulos do ref-gaussian

import numpy as np
import torch
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel, render_surfel
from scene import Scene
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips

AQUI = os.path.dirname(os.path.abspath(__file__))
COMBINACOES = [(30000, 30000), (40000, 40000), (50000, 48000), (50000, 50000)]


def avaliar(dataset, pipe, opt, it_pc, it_mesh):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=it_pc, shuffle=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    opt.indirect = 1
    gaussians.load_mesh_from_ply(dataset.model_path, it_mesh)
    ps, ss, ls = [], [], []
    for view in scene.getTestCameras():
        view.refl_mask = None
        r = render_surfel(view, gaussians, pipe, bg, srgb=opt.srgb, opt=opt)
        img = torch.clamp(r["render"], 0, 1)[None]
        gt = torch.clamp(view.original_image, 0, 1)[None, 0:3]
        ps.append(psnr(img, gt).item())
        ss.append(ssim(img, gt).item())
        ls.append(lpips(img, gt, net_type="vgg").item())
    del scene, gaussians
    torch.cuda.empty_cache()
    return {"gaussianas": it_pc, "malha": it_mesh, "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
            "lpips": float(np.mean(ls)), "psnr_por_vista": ps}


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    cena = os.path.basename(os.path.normpath(dataset.model_path))
    resultados = []
    with torch.no_grad():
        for it_pc, it_mesh in COMBINACOES:
            malha = os.path.join(dataset.model_path, f"test_{it_mesh:06d}.ply")
            if not os.path.exists(malha):
                print(f"pulando ({it_pc}, {it_mesh}): malha nao existe")
                continue
            r = avaliar(dataset, pipe, op, it_pc, it_mesh)
            resultados.append(r)
            print(f"{cena}  gaussianas {it_pc:>6}  malha {it_mesh:>6}  PSNR {r['psnr']:.2f}  SSIM {r['ssim']:.3f}  LPIPS {r['lpips']:.3f}", flush=True)
    json.dump({"cena": cena, "resultados": resultados}, open(os.path.join(AQUI, f"{cena}_checkpoints.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
