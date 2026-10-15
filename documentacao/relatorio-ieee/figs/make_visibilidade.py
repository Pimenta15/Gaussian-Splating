"""Testa a hipotese da visibilidade: exporta V, luz direta e indireta por vista de teste.

Rodar a partir da pasta ref-gaussian, no ambiente do compilador:
  run_ref_gaussian.bat ..\\documentacao\\relatorio-ieee\\figs\\make_visibilidade.py -m output/toaster --white_background

Gera em figs/:
  <cena>_visibilidade.png        pior e melhor vista: real, gerada, visibilidade, luz direta, luz indireta, erro
  <cena>_vis_vs_psnr.png         fracao de pixels ocluidos (V=0) x PSNR, por vista
  <cena>_visibilidade.json       estatisticas por vista
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
from utils.system_utils import searchForMaxIteration

AQUI = os.path.dirname(os.path.abspath(__file__))


def psnr(a, b):
    return float(10 * torch.log10(1.0 / torch.mean((a - b) ** 2)))


def para_img(t):
    """tensor (C,H,W) em [0,1] -> uint8 HxWx3"""
    t = t.detach().clamp(0, 1)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    return (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    dataset, pipe, opt = model.extract(args), pipeline.extract(args), op
    cena = os.path.basename(os.path.normpath(dataset.model_path))

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        it = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
        opt.indirect = 1
        gaussians.load_mesh_from_ply(dataset.model_path, it)

        stats, guardados = [], {}
        views = scene.getTestCameras()
        for i, view in enumerate(views):
            view.refl_mask = None
            r = render_surfel(view, gaussians, pipe, bg, srgb=opt.srgb, opt=opt)
            img = torch.clamp(r["render"], 0, 1)
            gt = torch.clamp(view.original_image, 0, 1)[0:3]
            alpha = r["rend_alpha"][0]
            obj = alpha > 0.5
            vis = r["visibility"][0]
            err = (img - gt).abs().mean(0)
            ocl = obj & (vis < 0.5)
            refl = r["refl_strength_map"][0]
            s = {
                "vista": i,
                "psnr": psnr(img, gt),
                "frac_ocluida": float(ocl.sum() / obj.sum().clamp_min(1)),
                "erro_ocluido": float(err[ocl].mean()) if ocl.any() else None,
                "erro_visivel": float(err[obj & ~ocl].mean()),
                "metal_medio_ocluido": float(refl[ocl].mean()) if ocl.any() else None,
            }
            stats.append(s)
            guardados[i] = None  # so guarda tensores das vistas escolhidas depois
            print(f"vista {i:3d}  psnr {s['psnr']:.2f}  ocluida {100*s['frac_ocluida']:.1f}%", flush=True)

        ps = np.array([s["psnr"] for s in stats])
        fr = np.array([s["frac_ocluida"] for s in stats])
        corr = float(np.corrcoef(fr, ps)[0, 1])
        imin, imax = int(ps.argmin()), int(ps.argmax())

        # paineis da pior e da melhor vista
        from PIL import Image, ImageDraw, ImageFont
        try:
            fonte = ImageFont.truetype("arial.ttf", 26)
        except OSError:
            fonte = ImageFont.load_default()
        linhas = []
        for i in (imin, imax):
            view = views[i]
            r = render_surfel(view, gaussians, pipe, bg, srgb=opt.srgb, opt=opt)
            img = torch.clamp(r["render"], 0, 1)
            gt = torch.clamp(view.original_image, 0, 1)[0:3]
            err = ((img - gt).abs().mean(0, keepdim=True) * 4).clamp(0, 1)
            alpha = r["rend_alpha"]
            vis = r["visibility"] * alpha + (1 - alpha) * 0.5  # fundo em cinza
            paineis = [(para_img(gt), f"Real (vista {i})"), (para_img(img), f"Gerada - {stats[i]['psnr']:.1f} dB"),
                       (para_img(vis), "Visibilidade V"), (para_img(r["direct_light"]), "Luz direta (ambiente)"),
                       (para_img(r["indirect_light"]), "Luz indireta (objeto)"), (para_img(err), "Erro (4x)")]
            w = 360
            faixa = 40
            linha = Image.new("RGB", (w * len(paineis), w + faixa), "white")
            d = ImageDraw.Draw(linha)
            for k, (arr, tit) in enumerate(paineis):
                linha.paste(Image.fromarray(arr).resize((w, w)), (k * w, faixa))
                tw = d.textlength(tit, font=fonte)
                d.text((k * w + (w - tw) / 2, 6), tit, fill="black", font=fonte)
            linhas.append(linha)
        out = Image.new("RGB", (linhas[0].width, sum(l.height for l in linhas)), "white")
        y = 0
        for l in linhas:
            out.paste(l, (0, y))
            y += l.height
        out.save(os.path.join(AQUI, f"{cena}_visibilidade.png"))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(3.5, 2.4), dpi=200)
        plt.scatter(100 * fr, ps, s=6, alpha=0.7)
        plt.xlabel("pixels do objeto com V = 0 (%)", fontsize=7)
        plt.ylabel("PSNR da vista (dB)", fontsize=7)
        plt.title(f"r = {corr:.2f}", fontsize=7)
        plt.tick_params(labelsize=6)
        plt.grid(alpha=0.3, lw=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(AQUI, f"{cena}_vis_vs_psnr.png"))

        def media(chave, sel):
            v = [s[chave] for s in sel if s[chave] is not None]
            return float(np.mean(v)) if v else None

        piores = sorted(stats, key=lambda s: s["psnr"])[:20]
        melhores = sorted(stats, key=lambda s: s["psnr"])[-20:]
        resumo = {
            "cena": cena,
            "vistas": len(stats),
            "correlacao_ocluida_psnr": corr,
            "pior": stats[imin], "melhor": stats[imax],
            "frac_ocluida_media": float(fr.mean()),
            "frac_ocluida_20_piores": media("frac_ocluida", piores),
            "frac_ocluida_20_melhores": media("frac_ocluida", melhores),
            "erro_ocluido_medio": media("erro_ocluido", stats),
            "erro_visivel_medio": media("erro_visivel", stats),
            "por_vista": stats,
        }
        json.dump(resumo, open(os.path.join(AQUI, f"{cena}_visibilidade.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in resumo.items() if k != "por_vista"}, indent=1))


if __name__ == "__main__":
    main()
