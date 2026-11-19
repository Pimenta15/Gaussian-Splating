"""Que parte do ambiente refletido nas vistas de avaliação nunca apareceu refletida no treino?

Só usa a geometria de referência do Shiny Blender (normais e alfa verdadeiros) e as câmeras. Para cada pixel do
objeto, a direção refletida é r = d - 2 (d·n) n, onde d é o raio da câmera e n a normal verdadeira. As direções
refletidas das vistas de treino (abaixo da faixa) marcam células cobertas numa grade da esfera; para cada vista de
avaliação, medimos a fração dos pixels do objeto cuja direção refletida cai numa célula que o treino nunca viu.
Um modelo que calcula o reflexo a partir de um mapa de ambiente (Ref-Gaussian) só sabe o ambiente nas células
cobertas, mesmo com a geometria perfeita.

    python cobertura_reflexo.py <ref_nerf> <saida.json> [banda=25] [passo_grade=5]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CENAS = ["ball", "car", "coffee", "helmet", "teapot", "toaster"]


def elev(m: np.ndarray) -> float:
    t = m[:3, 3]
    return math.degrees(math.asin(t[2] / np.linalg.norm(t)))


def direcoes(pasta: Path, nome: str, m: np.ndarray, fov_x: float, passo: int = 4) -> np.ndarray:
    """Direções refletidas (mundo) nos pixels do objeto, subamostrados a cada `passo` pixels."""
    img = np.asarray(Image.open(pasta / f"{nome}.png"))
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[..., 3]
    else:
        a = np.asarray(Image.open(pasta / f"{nome}_alpha.png"))
        a = a[..., 0] if a.ndim == 3 else a
    n = np.asarray(Image.open(pasta / f"{nome}_normal.png"))[..., :3].astype(np.float32) / 255 * 2 - 1
    h, w = a.shape
    f = 0.5 * w / math.tan(0.5 * fov_x)
    v, u = np.mgrid[0:h:passo, 0:w:passo]
    sel = a[v, u] > 127
    u, v = u[sel] + 0.5, v[sel] + 0.5
    nn = n[v.astype(int), u.astype(int)]
    nn /= np.linalg.norm(nn, axis=-1, keepdims=True).clip(1e-6)
    d_cam = np.stack([(u - w / 2) / f, -(v - h / 2) / f, -np.ones_like(u)], -1)  # câmera do Blender (OpenGL)
    d = d_cam @ m[:3, :3].T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    r = d - 2 * (d * nn).sum(-1, keepdims=True) * nn
    return r / np.linalg.norm(r, axis=-1, keepdims=True).clip(1e-6)


def celula(r: np.ndarray, passo_grade: float) -> np.ndarray:
    """Índice da célula numa grade de área igual: faixas de sen(elevação) e de azimute."""
    nz = int(round(180 / passo_grade))
    na = int(round(360 / passo_grade))
    iz = np.clip(((r[:, 2] + 1) / 2 * nz).astype(int), 0, nz - 1)
    ia = np.clip(((np.arctan2(r[:, 1], r[:, 0]) + math.pi) / (2 * math.pi) * na).astype(int), 0, na - 1)
    return iz * na + ia


def main():
    ref, saida = Path(sys.argv[1]), Path(sys.argv[2])
    banda = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
    passo_grade = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    out = {}
    for c in CENAS:
        tr = json.loads((ref / c / "transforms_train.json").read_text())
        te = json.loads((ref / c / "transforms_test.json").read_text())
        fov = tr["camera_angle_x"]
        ncel = int(round(180 / passo_grade)) * int(round(360 / passo_grade))
        contagem = np.zeros(ncel)
        n_treino = 0
        for fr in tr["frames"]:
            m = np.array(fr["transform_matrix"], dtype=np.float64)
            if 0 <= elev(m) < banda:  # mesma regra de rodar.py
                n_treino += 1
                cel = celula(direcoes(ref / c / "train", Path(fr["file_path"]).name, m, fov), passo_grade)
                contagem += np.bincount(cel, minlength=ncel)
        coberto = np.flatnonzero(contagem > 0)
        densidade = {"faixa": [], "acima": [], "cima": []}
        grupos = {"faixa": [], "acima": [], "cima": []}
        vistas = [("test", fr) for fr in te["frames"]] + [("train", fr) for fr in tr["frames"]]
        for pasta, fr in vistas:
            m = np.array(fr["transform_matrix"], dtype=np.float64)
            e = elev(m)
            if pasta == "test":
                g = "faixa" if e < banda else "acima"
            elif e >= 50:
                g = "cima"
            else:
                continue
            cel = celula(direcoes(ref / c / pasta, Path(fr["file_path"]).name, m, fov), passo_grade)
            grupos[g].append(float(1 - np.isin(cel, coberto).mean()))
            densidade[g].append(float(np.median(contagem[cel])))  # amostras de treino na célula do pixel típico
        out[c] = {"n_treino": n_treino, "celulas_cobertas": int(len(coberto)),
                  **{g: {"n": len(v), "frac_nao_vista": float(np.mean(v)),
                         "densidade_mediana": float(np.median(densidade[g]))} for g, v in grupos.items()}}
        print(c, n_treino, len(coberto), {g: (round(float(np.mean(v)), 3), round(float(np.median(densidade[g])), 1))
                                          for g, v in grupos.items()}, flush=True)
    saida.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
