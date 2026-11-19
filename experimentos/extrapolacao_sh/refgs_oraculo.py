"""Oráculos no Ref-Gaussian: por que ele perde de cima, mesmo com a geometria certa?

Partindo do modelo treinado só abaixo da faixa, reajusta UMA parte da aparência com vistas de referência de cima
(índices pares do grupo "de cima") e avalia nas vistas de cima de índice ímpar, que ficam fora do ajuste.
Geometria (posições, escalas, rotações, opacidades, normais) fica congelada em todos os casos.

  material   cor base, cor difusa, força do reflexo, rugosidade e metalicidade de cada Gaussiana
  ambiente   o mapa de ambiente compartilhado pela cena
  indireta   a luz indireta aprendida por Gaussiana (SH + ASG), usada onde o raio refletido é bloqueado
  aparencia  as três partes juntas (teto do que a aparência sozinha consegue)

Se o ajuste só do material recuperar quase tudo o que a aparência inteira recupera, o que falta de cima é o
material visto de ângulos novos; se for o ambiente, falta iluminação; se for a indireta, falta a luz refletida
pelo próprio objeto.

Roda dentro do contêiner, a partir do código oficial do Ref-Gaussian:
    python /exp/refgs_oraculo.py -m /exp/saidas/car/refgs_b25 --passos 1000
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

REFGS = os.environ.get("REFGS_DIR", "/opt/ref-gaussian")
sys.path.insert(0, REFGS)
os.chdir(REFGS)  # o código oficial lê arquivos auxiliares por caminho relativo

import torch  # noqa: E402
from argparse import ArgumentParser  # noqa: E402
from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import GaussianModel, render_surfel  # noqa: E402
from lpipsPyTorch import lpips  # noqa: E402
from scene import Scene  # noqa: E402
from utils.image_utils import psnr  # noqa: E402
from utils.loss_utils import l1_loss, ssim  # noqa: E402
from utils.system_utils import searchForMaxIteration  # noqa: E402


def elevacao(fr) -> float:
    t = [r[3] for r in fr["transform_matrix"][:3]]
    return math.degrees(math.asin(t[2] / math.sqrt(sum(x * x for x in t))))


def grupos_de_parametros(g: GaussianModel, op, extensao: float) -> dict:
    material = [(g._refl_strength, op.refl_strength_lr), (g._ori_color, op.ori_color_lr),
                (g._diffuse_color, op.ori_color_lr), (g._roughness, op.roughness_lr), (g._metalness, op.metalness_lr)]
    ambiente = [(p, op.envmap_cubemap_lr) for p in g.env_map.parameters()]
    indireta = [(g._indirect_dc, op.indirect_lr), (g._indirect_rest, op.indirect_lr / 20.0),
                (g._indirect_asg, op.asg_lr)]
    # geometria: posição (taxa do treino, escalada pela extensão da cena), forma, opacidade e normais
    geometria = [(g._xyz, op.position_lr_init * extensao), (g._scaling, op.scaling_lr), (g._rotation, op.rotation_lr),
                 (g._opacity, op.opacity_lr), (g._normal1, op.normal_lr), (g._normal2, op.normal_lr)]
    return {"material": material, "ambiente": ambiente, "indireta": indireta,
            "aparencia": material + ambiente + indireta, "geometria": geometria,
            "tudo": material + ambiente + indireta + geometria}


def carregar(dataset, op, iteracao):
    g = GaussianModel(dataset.sh_degree)
    cena = Scene(dataset, g, load_iteration=iteracao, shuffle=False)
    op.indirect = 1
    g.load_mesh_from_ply(dataset.model_path, searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud")))
    return g, cena


def avaliar(g, cams, pipe, fundo, op) -> dict:
    ps, lp = [], []
    with torch.no_grad():
        g.env_map.build_mips()
        for v in cams:
            v.refl_mask = None
            img = torch.clamp(render_surfel(v, g, pipe, fundo, srgb=op.srgb, opt=op)["render"], 0, 1)[None]
            gt = torch.clamp(v.original_image.cuda(), 0, 1)[None, :3]
            ps.append(psnr(img, gt).item())
            lp.append(lpips(img, gt, net_type="vgg").item())
    return {"psnr": sum(ps) / len(ps), "lpips": sum(lp) / len(lp), "n": len(ps)}


def main():
    ap = ArgumentParser()
    modelo = ModelParams(ap, sentinel=True)
    op_p = OptimizationParams(ap)
    pipe_p = PipelineParams(ap)
    ap.add_argument("--iteration", default=-1, type=int)
    ap.add_argument("--passos", default=1000, type=int)
    ap.add_argument("--grupos", default="material,ambiente,indireta,aparencia,geometria,tudo")
    args = get_combined_args(ap)
    args.data_device = "cpu"  # 263 imagens na GPU de 6 GB não cabem junto com o modelo; vão uma a uma
    dataset, pipe, op = modelo.extract(args), pipe_p.extract(args), op_p
    for k, v in vars(args).items():  # o OptimizationParams é passado como no eval.py oficial
        if hasattr(op, k):
            setattr(op, k, v)
    fundo = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    frames = json.loads((Path(dataset.source_path) / "transforms_test.json").read_text())["frames"]
    cima = [i for i, fr in enumerate(frames) if fr.get("origem", "teste") != "teste" and elevacao(fr) >= 50]
    pares, impares = cima[0::2], cima[1::2]
    saida = Path(dataset.model_path) / "oraculo_refgs.json"
    res = json.loads(saida.read_text()) if saida.exists() else {}

    g, cena = carregar(dataset, op, args.iteration)
    cams = cena.getTestCameras()
    if "original" not in res:
        res["original"] = {"impares": avaliar(g, [cams[i] for i in impares], pipe, fundo, op),
                           "pares": avaliar(g, [cams[i] for i in pares], pipe, fundo, op)}
        print("original", json.dumps(res["original"]), flush=True)
    for nome in args.grupos.split(","):
        if nome in res:
            continue
        del g, cena  # libera o modelo anterior antes de carregar outro (a GPU tem 6 GB)
        torch.cuda.empty_cache()
        g, cena = carregar(dataset, op, args.iteration)
        cams = cena.getTestCameras()
        livres = grupos_de_parametros(g, op, cena.cameras_extent)[nome]
        ids_livres = {id(p) for p, _ in livres}
        todos = [p for p in vars(g).values() if isinstance(p, torch.nn.Parameter)]
        todos += list(g.env_map.parameters()) + list(g.env_map_2.parameters())
        for p in todos:
            p.requires_grad_(id(p) in ids_livres)
        ajusta_ambiente = nome in ("ambiente", "aparencia", "tudo")
        if not ajusta_ambiente:  # os níveis do mapa foram montados no carregamento, ainda presos ao grafo
            with torch.no_grad():
                g.env_map.build_mips()
        otim = torch.optim.Adam([{"params": [p], "lr": lr} for p, lr in livres], eps=1e-15)
        random.seed(0)
        t0 = time.time()
        for passo in range(args.passos):
            v = cams[random.choice(pares)]
            v.refl_mask = None
            if ajusta_ambiente:
                g.env_map.build_mips()
            img = render_surfel(v, g, pipe, fundo, srgb=op.srgb, opt=op)["render"]
            gt = v.original_image[:3].cuda()
            perda = 0.8 * l1_loss(img, gt) + 0.2 * (1.0 - ssim(img, gt))
            perda.backward()
            otim.step()
            otim.zero_grad(set_to_none=True)
        res[nome] = {"impares": avaliar(g, [cams[i] for i in impares], pipe, fundo, op),
                     "pares": avaliar(g, [cams[i] for i in pares], pipe, fundo, op),
                     "passos": args.passos, "tempo_s": time.time() - t0}
        print(nome, json.dumps(res[nome]), flush=True)
        saida.write_text(json.dumps(res, indent=1))
        del otim
        torch.cuda.empty_cache()
    saida.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
