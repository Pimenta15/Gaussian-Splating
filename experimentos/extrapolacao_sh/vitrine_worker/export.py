"""Exportação de uma cena de Gaussianas para o formato .splat usado pelos visualizadores web.

Formato (32 bytes por Gaussiana, little-endian): posição float32x3, escala float32x3 (já em
espaço linear), cor RGBA uint8x4 (cor difusa vinda do harmônico de grau 0 e opacidade) e rotação
uint8x4 (quaternion normalizado w, x, y, z mapeado de [-1, 1] para [0, 255]).
As Gaussianas são ordenadas da mais "importante" (volume x opacidade) para a menos.
"""
from __future__ import annotations

import numpy as np

SH_C0 = 0.28209479177387814  # constante do harmônico esférico de grau 0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def para_splat(
    means: np.ndarray,
    log_scales: np.ndarray,
    quats: np.ndarray,
    logit_opacities: np.ndarray,
    sh0: np.ndarray,
) -> bytes:
    """Converte parâmetros brutos de 3DGS (como otimizados) em bytes .splat.

    means [N,3], log_scales [N,3], quats [N,4] (w,x,y,z, não precisam estar normalizados),
    logit_opacities [N] ou [N,1], sh0 [N,3] ou [N,1,3].
    """
    n = means.shape[0]
    scales = np.exp(log_scales.reshape(n, 3)).astype(np.float32)
    opac = _sigmoid(logit_opacities.reshape(n)).astype(np.float32)
    cor = np.clip(0.5 + SH_C0 * sh0.reshape(n, 3), 0.0, 1.0)

    q = quats.reshape(n, 4).astype(np.float64)
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)

    ordem = np.argsort(-(scales.prod(axis=1) * opac))

    reg = np.zeros(n, dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    reg["pos"] = means.reshape(n, 3)[ordem]
    reg["scale"] = scales[ordem]
    reg["rgba"][:, :3] = np.round(cor[ordem] * 255)
    reg["rgba"][:, 3] = np.round(opac[ordem] * 255)
    reg["rot"] = np.clip(np.round(q[ordem] * 128 + 128), 0, 255)
    return reg.tobytes()


def ler_splat(dados: bytes) -> dict[str, np.ndarray]:
    """Inverso aproximado de `para_splat` (útil para testes e para inspecionar arquivos)."""
    reg = np.frombuffer(dados, dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    return {
        "means": reg["pos"].copy(),
        "scales": reg["scale"].copy(),
        "rgba": reg["rgba"].astype(np.float32) / 255.0,
        "quats": (reg["rot"].astype(np.float32) - 128.0) / 128.0,
    }
