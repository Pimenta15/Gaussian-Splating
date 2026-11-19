"""Renderiza, nas vistas da varredura de elevação da demonstração, os modelos treinados com TODAS as alturas.

  3DGS completo          gsplat, 30 mil iterações, 100 vistas de treino (saidas/<cena>/completo_sh3/params.pt)
  Ref-Gaussian completo  código oficial, 50 mil iterações, 100 vistas (ref-gaussian/output/<cena>)

Mede o PSNR em resolução cheia contra a referência e marca se a vista fez parte do treino desses modelos (as vistas
de cima da varredura são vistas de treino deles; as de teste não). Grava explorar/<i>_refgs_completo.jpg e
explorar/modelos.json.

Roda no contêiner com o código do Ref-Gaussian em /opt/ref-gaussian, ref-gaussian/output montado em
/opt/ref-gaussian/output e ref_nerf montado em /opt/ref-gaussian/data/ref_nerf e em /data:
    python /exp/demo_modelos.py --cena car
"""
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, "/exp")
REFGS = "/opt/ref-gaussian"
sys.path.insert(1, REFGS)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def elev_az(m):
    t = [r[3] for r in m[:3]]
    n = math.sqrt(sum(x * x for x in t))
    return math.degrees(math.asin(t[2] / n)), math.degrees(math.atan2(t[1], t[0]))


def psnr_np(a, b):
    return float(-10 * math.log10(max(float(((a - b) ** 2).mean()), 1e-12)))


def main():
    cena = sys.argv[sys.argv.index("--cena") + 1]
    pasta = Path("/exp/saidas") / cena / "b25_sh3" / "explorar"
    meta = json.loads((pasta / "meta.json").read_text())
    dados = Path("/data") / cena
    frames = {"teste": json.loads((dados / "transforms_test.json").read_text())["frames"],
              "treino_nao_usada": json.loads((dados / "transforms_train.json").read_text())["frames"]}
    nomes = []
    for m in meta:  # identifica cada vista da varredura pelo par (elevação, azimute)
        cand = [Path(fr["file_path"]).name for fr in frames[m["origem"]]
                if all(abs(x - y) < 0.05 for x, y in zip(elev_az(fr["transform_matrix"]), (m["elev"], m["azim"])))]
        nomes.append(cand[0] if len(cand) == 1 else None)

    out = []
    # 3DGS treinado com todas as alturas
    from rodar import renderizar  # noqa: PLC0415
    from vitrine_worker.datasets import ler_transforms  # noqa: PLC0415
    p3 = Path("/exp/saidas") / cena / "completo_sh3" / "params.pt"
    params = {k: v.cuda() for k, v in torch.load(p3).items()} if p3.exists() else None
    cams3 = {("teste", c.nome): c for c in ler_transforms(dados / "transforms_test.json", fundo_branco=True)}
    cams3.update({("treino_nao_usada", c.nome): c for c in ler_transforms(dados / "transforms_train.json",
                                                                           fundo_branco=True)})
    for m, nome in zip(meta, nomes):
        lin = {"i": m["i"], "nome": nome, "treino_dos_completos": m["origem"] != "teste"}
        c = cams3.get((m["origem"], nome))
        if params is not None and c is not None:
            img = renderizar(params, c).cpu().numpy()
            lin["psnr_completo"] = psnr_np(img, c.imagem)
        out.append(lin)
    del params
    torch.cuda.empty_cache()

    # Ref-Gaussian treinado com todas as alturas (replicação do artigo)
    os.chdir(REFGS)
    from argparse import ArgumentParser  # noqa: PLC0415
    from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args  # noqa: PLC0415
    from gaussian_renderer import GaussianModel, render_surfel  # noqa: PLC0415
    from scene import Scene  # noqa: PLC0415
    from utils.system_utils import searchForMaxIteration  # noqa: PLC0415
    sys.argv = [sys.argv[0], "-m", f"output/{cena}", "-s", f"data/ref_nerf/{cena}"]
    ap = ArgumentParser()
    mp, op, pp = ModelParams(ap, sentinel=True), OptimizationParams(ap), PipelineParams(ap)
    ap.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(ap)
    args.data_device = "cpu"
    ds, pipe = mp.extract(args), pp.extract(args)
    with torch.no_grad():
        g = GaussianModel(ds.sh_degree)
        sc = Scene(ds, g, load_iteration=-1, shuffle=False)
        op.indirect = 1
        g.load_mesh_from_ply(ds.model_path, searchForMaxIteration(os.path.join(ds.model_path, "point_cloud")))
        g.env_map.build_mips()
        fundo = torch.tensor([1, 1, 1] if ds.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        cams = {("teste", v.image_name): v for v in sc.getTestCameras()}
        cams.update({("treino_nao_usada", v.image_name): v for v in sc.getTrainCameras()})
        for m, nome, lin in zip(meta, nomes, out):
            v = cams.get((m["origem"], nome))
            if v is None:
                continue
            v.refl_mask = None
            img = torch.clamp(render_surfel(v, g, pipe, fundo, srgb=op.srgb, opt=op)["render"], 0, 1)
            gt = torch.clamp(v.original_image.cuda(), 0, 1)[:3]
            lin["psnr_refgs_completo"] = float(-10 * torch.log10(((img - gt) ** 2).mean()).item())
            arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
            arr = cv2.resize(arr, (400, 400), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(pasta / f"{m['i']:02d}_refgs_completo.jpg"), arr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    (pasta / "modelos.json").write_text(json.dumps(out, indent=1))
    print(cena, sum("psnr_refgs_completo" in o for o in out), "vistas do Ref-Gaussian completo,",
          sum("psnr_completo" in o for o in out), "do 3DGS completo", flush=True)


if __name__ == "__main__":
    main()
