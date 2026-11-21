"""Remove dos harmônicos esféricos a parte que nenhuma câmera observou (sem treino).

A cor de cada gaussiana depende do ângulo de visão por harmônicos esféricos (SH). Se o vídeo só
filmou o carro de lado (todas as câmeras perto da horizontal), os termos do SH que dependem da
altura do olhar nunca foram ajustados e ficam com valores arbitrários: vistas de cima mostram
manchas de cor e reflexos inventados.

Aqui, para cada gaussiana, os coeficientes c são trocados pela solução de norma mínima que
reproduz a mesma cor em todas as direções observadas D:

    c' = argmin ||B_D c' - B_D c||^2 + ||L c'||^2   =>   c' = (B_D^T B_D + L^T L)^-1 B_D^T B_D c

com L penalizando só os termos não constantes. Nas direções observadas a cor praticamente não
muda; fora delas, o SH tende à cor média. Como B_D é igual para todas as gaussianas, é uma única
matriz 16x16 aplicada a todas: segundos, sem rede neural, vale para qualquer material.
"""
from __future__ import annotations

import numpy as np

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def base_sh(d: np.ndarray, grau: int = 3) -> np.ndarray:
    """Base dos SH na mesma convenção do 3DGS/gsplat. d: [N,3] unitárias -> [N,(grau+1)^2]."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    cols = [np.full_like(x, C0)]
    if grau >= 1:
        cols += [-C1 * y, C1 * z, -C1 * x]
    if grau >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        cols += [C2[0] * xy, C2[1] * yz, C2[2] * (2 * zz - xx - yy), C2[3] * xz, C2[4] * (xx - yy)]
    if grau >= 3:
        cols += [C3[0] * y * (3 * xx - yy), C3[1] * xy * z, C3[2] * y * (4 * zz - xx - yy),
                 C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
                 C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)]
    return np.stack(cols, 1)


def direcoes_observadas(elev_min: float, elev_max: float, margem: float = 5.0, n_az: int = 72,
                        n_el: int = 9) -> np.ndarray:
    """Direções de visão (câmera -> objeto) de uma volta completa entre as elevações filmadas (graus)."""
    az = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    el = np.radians(np.linspace(elev_min - margem, elev_max + margem, n_el))
    a, e = np.meshgrid(az, el)
    camera = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], -1).reshape(-1, 3)
    return -camera  # o gsplat avalia o SH na direção da câmera para a gaussiana


def matriz_projecao(direcoes: np.ndarray, grau: int = 3, lam: float = 1e-2) -> np.ndarray:
    """M tal que c' = M c (c com (grau+1)^2 coeficientes por canal)."""
    B = base_sh(direcoes, grau)
    G = B.T @ B / len(B)
    L = np.eye(G.shape[0]) * lam
    L[0, 0] = 0.0  # a cor média (termo constante) nunca é penalizada
    return np.linalg.solve(G + L, G)


def projetar(sh0: np.ndarray, shN: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sh0 [N,1,3] ou [N,3], shN [N,K-1,3] -> mesmos formatos, com o SH projetado."""
    n = sh0.shape[0]
    c = np.concatenate([sh0.reshape(n, 1, 3), shN], 1)  # [N,K,3]
    c2 = np.einsum("kj,njc->nkc", M, c)
    return c2[:, :1].reshape(sh0.shape), c2[:, 1:]


def ler_ply(caminho) -> tuple[list[str], np.ndarray]:
    """Lê um .ply binário de gaussianas (todas as propriedades float32), como o gravado pelo treino."""
    bruto = open(caminho, "rb").read()
    fim = bruto.index(b"end_header\n") + len(b"end_header\n")
    cab = bruto[:fim].decode().splitlines()
    n = int(next(ln.split()[2] for ln in cab if ln.startswith("element vertex")))
    nomes = [ln.split()[2] for ln in cab if ln.startswith("property float")]
    return nomes, np.frombuffer(bruto[fim:], dtype="<f4").reshape(n, len(nomes)).copy()


def gravar_ply(caminho, nomes: list[str], dados: np.ndarray) -> None:
    cab = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(dados)
    cab += "".join(f"property float {c}\n" for c in nomes) + "end_header\n"
    with open(caminho, "wb") as f:
        f.write(cab.encode())
        f.write(dados.astype("<f4").tobytes())


def projetar_ply(entrada, saida, elev_min: float, elev_max: float) -> int:
    """Aplica a projeção a um .ply de gaussianas (layout do 3DGS) e grava o resultado."""
    nomes, d = ler_ply(entrada)
    idx = {n: i for i, n in enumerate(nomes)}
    rest = sorted((n for n in nomes if n.startswith("f_rest_")), key=lambda n: int(n.split("_")[2]))
    k = len(rest) // 3  # coeficientes por canal, sem o constante
    grau = int(round(np.sqrt(k + 1))) - 1
    sh0 = d[:, [idx[f"f_dc_{i}"] for i in range(3)]].reshape(-1, 1, 3)
    # no .ply do 3DGS os f_rest vêm canal a canal: R(1..k), G(1..k), B(1..k)
    shN = d[:, [idx[n] for n in rest]].reshape(-1, 3, k).transpose(0, 2, 1)
    sh0b, shNb = projetar(sh0, shN, matriz_projecao(direcoes_observadas(elev_min, elev_max), grau))
    d[:, [idx[f"f_dc_{i}"] for i in range(3)]] = sh0b.reshape(-1, 3)
    d[:, [idx[n] for n in rest]] = shNb.transpose(0, 2, 1).reshape(-1, 3 * k)
    gravar_ply(saida, nomes, d)
    return len(d)


def _fibonacci(n: int = 2000) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def matriz_rotacao(R: np.ndarray, grau: int = 3) -> np.ndarray:
    """M tal que, girando o modelo por R (pontos p -> R p), os SH girados são c' = M c.

    A cor nova na direção d é a antiga na direção R^T d: B(d) c' = B(R^T d) c, resolvido por mínimos
    quadrados em direções espalhadas na esfera (exato, porque cada grau dos SH é fechado por rotação).
    """
    d = _fibonacci()
    return np.linalg.lstsq(base_sh(d, grau), base_sh(d @ R, grau), rcond=None)[0]
