"""Teste de autointersecao no raio de visibilidade do Ref-Gaussian.

No codigo original (utils/refl_utils.py, get_specular_color_surfel) o raio de visibilidade parte
exatamente do ponto da superficie, sem deslocamento. Aqui a mesma funcao e reproduzida com a
origem deslocada por eps * direcao_refletida e o modelo treinado e reavaliado para varios eps,
sem novo treino e sem alterar o codigo do projeto (a funcao e substituida so neste processo).

Rodar a partir da pasta ref-gaussian, no ambiente do compilador:
  run_ref_gaussian.bat ..\\documentacao\\relatorio-ieee\\figs\\teste_offset_visibilidade.py -m output/toaster --white_background
Saida: figs/<cena>_offset.json e figs/<cena>_offset_pior.png
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from argparse import ArgumentParser

import gaussian_renderer as gr
import utils.refl_utils as ru
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel, render_surfel
from scene import Scene
from utils.system_utils import searchForMaxIteration

AQUI = os.path.dirname(os.path.abspath(__file__))
EPS = [0.0, 0.002, 0.01, 0.05]
_eps = 0.0


def especular_com_offset(envmap, albedo, HWK, R, T, normal_map, render_alpha, scaling_modifier=1.0,
                         refl_strength=None, roughness=None, pc=None, surf_depth=None, indirect_light=None):
    """Copia de ru.get_specular_color_surfel; a unica mudanca e o deslocamento da origem do raio."""
    H, W, K = HWK
    rays_cam, rays_o = ru.sample_camera_rays(HWK, R, T)
    w_o = -rays_cam
    rays_refl, NdotV = ru.reflection(w_o, normal_map)
    rays_refl = ru.safe_normalize(rays_refl)
    fg_uv = torch.cat([NdotV, roughness], -1).clamp(0, 1)
    fg = ru.dr.texture(ru.FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(), filter_mode="linear",
                       boundary_mode="clamp").reshape(1, H, W, 2)
    direct_light = envmap(rays_refl, roughness=roughness)
    specular_weight = ((0.04 * (1 - refl_strength) + albedo * refl_strength) * fg[0][..., 0:1] + fg[0][..., 1:2])
    visibility = torch.ones_like(render_alpha)
    if pc.ray_tracer is not None and indirect_light is not None:
        mask = (render_alpha > 0)[..., 0]
        rays_cam, rays_o = ru.sample_camera_rays_unnormalize(HWK, R, T)
        w_o = ru.safe_normalize(-rays_cam)
        rays_refl, _ = ru.reflection(w_o, normal_map)
        rays_refl = ru.safe_normalize(rays_refl)
        intersections = rays_o + surf_depth.permute(1, 2, 0) * rays_cam
        intersections = intersections + _eps * rays_refl  # <- deslocamento da origem
        _, _, depth = pc.ray_tracer.trace(intersections[mask], rays_refl[mask])
        visibility[mask] = (depth >= 10).float().unsqueeze(-1)
        specular_light = direct_light * visibility + (1 - visibility) * indirect_light
        indirect_color = (1 - visibility) * indirect_light * render_alpha * specular_weight
    else:
        specular_light = direct_light
    specular = specular_light * render_alpha * specular_weight
    extra = None
    if indirect_light is not None:
        extra = {"visibility": visibility.permute(2, 0, 1), "indirect_light": indirect_light.permute(2, 0, 1),
                 "direct_light": direct_light.permute(2, 0, 1), "indirect_color": indirect_color.permute(2, 0, 1)}
    return specular.permute(2, 0, 1), extra


def main():
    global _eps
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    dataset, pipe, opt = model.extract(args), pipeline.extract(args), op
    cena = os.path.basename(os.path.normpath(dataset.model_path))
    gr.get_specular_color_surfel = especular_com_offset

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        it = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
        opt.indirect = 1
        gaussians.load_mesh_from_ply(dataset.model_path, it)
        views = scene.getTestCameras()
        print("raio da cena (extent):", scene.cameras_extent, flush=True)

        res = []
        for e in EPS:
            _eps = e
            ps, ocl = [], []
            for view in views:
                view.refl_mask = None
                r = render_surfel(view, gaussians, pipe, bg, srgb=opt.srgb, opt=opt)
                img = torch.clamp(r["render"], 0, 1)
                gt = torch.clamp(view.original_image, 0, 1)[0:3]
                ps.append(float(10 * torch.log10(1 / torch.mean((img - gt) ** 2))))
                obj = r["rend_alpha"][0] > 0.5
                ocl.append(float(((r["visibility"][0] < 0.5) & obj).sum() / obj.sum().clamp_min(1)))
            ps = np.array(ps)
            linha = {"eps": e, "psnr_medio": float(ps.mean()), "psnr_pior_vista": float(ps.min()),
                     "vista_pior": int(ps.argmin()), "psnr_vista_90": float(ps[90]) if len(ps) > 90 else None,
                     "frac_ocluida_media": float(np.mean(ocl)), "vistas_abaixo_25": int((ps < 25).sum())}
            res.append(linha)
            print(json.dumps(linha), flush=True)

        # painel da vista 90 com o eps original e o maior
        from PIL import Image, ImageDraw, ImageFont
        try:
            fonte = ImageFont.truetype("arial.ttf", 26)
        except OSError:
            fonte = ImageFont.load_default()
        paineis = []
        for e in (EPS[0], EPS[-1]):
            _eps = e
            v = views[90]
            r = render_surfel(v, gaussians, pipe, bg, srgb=opt.srgb, opt=opt)
            a = r["rend_alpha"]
            vis = (r["visibility"] * a + (1 - a) * 0.5).repeat(3, 1, 1)
            for t, tit in ((torch.clamp(r["render"], 0, 1), f"Gerada (eps={e})"), (vis, f"Visibilidade (eps={e})")):
                arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                paineis.append((Image.fromarray(arr).resize((360, 360)), tit))
        w, faixa = 360, 40
        out = Image.new("RGB", (w * len(paineis), w + faixa), "white")
        d = ImageDraw.Draw(out)
        for k, (im, tit) in enumerate(paineis):
            out.paste(im, (k * w, faixa))
            tw = d.textlength(tit, font=fonte)
            d.text((k * w + (w - tw) / 2, 6), tit, fill="black", font=fonte)
        out.save(os.path.join(AQUI, f"{cena}_offset_pior.png"))
        json.dump({"cena": cena, "extent": float(scene.cameras_extent), "resultados": res},
                  open(os.path.join(AQUI, f"{cena}_offset.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
