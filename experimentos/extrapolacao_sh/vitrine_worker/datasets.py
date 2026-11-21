"""Leitura de cenas no formato de câmeras do Blender/NeRF (transforms_*.json).

É o formato do Shiny Blender e o formato para o qual o pipeline converte as poses do COLMAP.
Convenção: `transform_matrix` é câmera->mundo no padrão OpenGL (y para cima, z para trás);
aqui ela é convertida para OpenCV (y para baixo, z para frente), que é o que o gsplat espera.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

OPENGL_PARA_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class Camera:
    nome: str
    imagem: np.ndarray        # HxWx3 float32 em [0, 1], já composta sobre o fundo
    K: np.ndarray             # 3x3
    c2w: np.ndarray           # 4x4, convenção OpenCV

    @property
    def largura(self) -> int:
        return self.imagem.shape[1]

    @property
    def altura(self) -> int:
        return self.imagem.shape[0]

    @property
    def viewmat(self) -> np.ndarray:
        """mundo->câmera (4x4), como o rasterizador espera."""
        return np.linalg.inv(self.c2w)


def _ler_imagem(caminho: Path, fundo: np.ndarray, reduzir: int) -> np.ndarray:
    img = cv2.imread(str(caminho), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(caminho)
    if reduzir > 1:
        img = cv2.resize(img, (img.shape[1] // reduzir, img.shape[0] // reduzir), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.shape[2] == 4:  # BGRA -> composição sobre o fundo
        cor, alfa = img[..., :3], img[..., 3:4]
        img = cor * alfa + fundo[::-1] * (1 - alfa)
    return np.ascontiguousarray(img[..., ::-1])  # BGR -> RGB


def ler_transforms(arquivo: Path, fundo_branco: bool = True, reduzir: int = 1) -> list[Camera]:
    """Lê um transforms_*.json e devolve as câmeras com as imagens carregadas."""
    arquivo = Path(arquivo)
    meta = json.loads(arquivo.read_text())
    fundo = np.ones(3, np.float32) if fundo_branco else np.zeros(3, np.float32)
    cams = []
    for fr in meta["frames"]:
        caminho = arquivo.parent / fr["file_path"]
        if caminho.suffix == "":
            caminho = caminho.with_suffix(".png")
        img = _ler_imagem(caminho, fundo, reduzir)
        h, w = img.shape[:2]
        if "fl_x" in meta:
            fx, fy = meta["fl_x"] / reduzir, meta.get("fl_y", meta["fl_x"]) / reduzir
            cx, cy = meta.get("cx", w * reduzir / 2) / reduzir, meta.get("cy", h * reduzir / 2) / reduzir
        else:
            fx = fy = 0.5 * w / math.tan(0.5 * meta["camera_angle_x"])
            cx, cy = w / 2, h / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        c2w = np.array(fr["transform_matrix"], dtype=np.float64) @ OPENGL_PARA_OPENCV
        cams.append(Camera(Path(fr["file_path"]).name, img, K, c2w))
    return cams


def separar_validacao(cams: list[Camera], cada: int = 8) -> tuple[list[Camera], list[Camera]]:
    """Separa 1 a cada `cada` vistas para validação (quality gate), como no protocolo do 3DGS."""
    val = [c for i, c in enumerate(cams) if i % cada == 0]
    treino = [c for i, c in enumerate(cams) if i % cada != 0]
    return treino, val


def raio_da_cena(cams: list[Camera]) -> float:
    """Raio aproximado da cena: distância máxima dos centros das câmeras ao centro médio."""
    centros = np.stack([c.c2w[:3, 3] for c in cams])
    return float(np.linalg.norm(centros - centros.mean(0), axis=1).max() * 1.1)


def foco_das_cameras(cams: list[Camera]) -> tuple[np.ndarray, float]:
    """Ponto para onde as câmeras olham (mínimos quadrados dos eixos ópticos) e distância mediana até ele.

    Numa captura de produto as câmeras giram em volta do objeto, então esse ponto é o centro do objeto.
    """
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c in cams:
        o, d = c.c2w[:3, 3], c.c2w[:3, 2]  # centro e eixo z (para frente, OpenCV)
        P = np.eye(3) - np.outer(d, d)     # projeta no plano perpendicular ao raio
        A += P
        b += P @ o
    foco = np.linalg.lstsq(A, b, rcond=None)[0]
    dist = float(np.median([np.linalg.norm(c.c2w[:3, 3] - foco) for c in cams]))
    return foco, dist
