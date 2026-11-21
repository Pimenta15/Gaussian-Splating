"""Métricas de qualidade de imagem usadas no quality gate (PSNR, SSIM e LPIPS)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(img: torch.Tensor, ref: torch.Tensor) -> float:
    """img e ref: [H,W,3] ou [3,H,W] em [0, 1]."""
    mse = torch.mean((img.clamp(0, 1) - ref.clamp(0, 1)) ** 2)
    return float(10 * torch.log10(1.0 / mse.clamp_min(1e-12)))


def _janela(tam: int, sigma: float, canais: int, device) -> torch.Tensor:
    x = torch.arange(tam, dtype=torch.float32, device=device) - tam // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(1)
    w2 = (g @ g.t()).expand(canais, 1, tam, tam).contiguous()
    return w2


def ssim(img: torch.Tensor, ref: torch.Tensor, tam: int = 11) -> torch.Tensor:
    """SSIM com janela gaussiana (mesma definição usada no 3DGS). Entradas [B,3,H,W] em [0, 1]."""
    c = img.shape[1]
    w = _janela(tam, 1.5, c, img.device)
    mu1 = F.conv2d(img, w, padding=tam // 2, groups=c)
    mu2 = F.conv2d(ref, w, padding=tam // 2, groups=c)
    s11 = F.conv2d(img * img, w, padding=tam // 2, groups=c) - mu1**2
    s22 = F.conv2d(ref * ref, w, padding=tam // 2, groups=c) - mu2**2
    s12 = F.conv2d(img * ref, w, padding=tam // 2, groups=c) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    mapa = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1**2 + mu2**2 + c1) * (s11 + s22 + c2))
    return mapa.mean()


class LPIPS:
    """Wrapper preguiçoso: carrega a rede só se for usada (pesos baixados na primeira vez)."""

    def __init__(self, device: str = "cuda"):
        self._modelo = None
        self.device = device

    def __call__(self, img: torch.Tensor, ref: torch.Tensor) -> float:
        if self._modelo is None:
            import lpips  # noqa: PLC0415 - import pesado, só quando necessário

            self._modelo = lpips.LPIPS(net="vgg", verbose=False).to(self.device).eval()
        with torch.no_grad():
            return float(self._modelo(img * 2 - 1, ref * 2 - 1).mean())
