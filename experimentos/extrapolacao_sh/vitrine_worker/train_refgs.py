"""Rota premium: Ref-Gaussian (código oficial em /opt/ref-gaussian, commit fixo no Dockerfile).

O treino e a avaliação rodam como subprocessos dos scripts originais (train.py / eval.py), para não
alterar o método que está sendo replicado; este módulo só monta os argumentos, acompanha o progresso
e lê as métricas que o eval.py grava em metric.txt.
"""
from __future__ import annotations

import codecs
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REFGS_DIR = Path(os.getenv("REFGS_DIR", "/opt/ref-gaussian"))

# Parâmetros do cronograma do Ref-Gaussian ajustados para 50k iterações (arguments/__init__.py).
# No perfil reduzido todos são escalados juntos para manter as fases do método na mesma proporção.
CRONOGRAMA_50K = {
    "iterations": 50_000,
    "position_lr_max_steps": 30_000,
    "densify_until_iter": 25_000,
    "volume_render_until_iter": 18_000,
    "normal_smooth_until_iter": 18_000,
    "indirect_from_iter": 20_000,
    "feature_rest_from_iter": 5_000,
    "normal_prop_until_iter": 25_000,
    "dist_loss_start": 3_000,
}


@dataclass
class ConfigRefGS:
    iteracoes: int = 50_000
    fundo_branco: bool = True


INTERVALO_MALHA = 2000  # MESH_EXTRACT_INTERVAL do train.py


def argumentos_treino(dados: Path, saida: Path, cfg: ConfigRefGS) -> list[str]:
    # o eval.py carrega a malha test_<última iteração>.ply, que só existe se a iteração final
    # cair na extração periódica; falhar aqui evita perder o treino inteiro na avaliação
    if cfg.iteracoes % INTERVALO_MALHA:
        raise ValueError(f"iteracoes do Ref-Gaussian deve ser múltiplo de {INTERVALO_MALHA} (recebido {cfg.iteracoes})")
    escala = cfg.iteracoes / CRONOGRAMA_50K["iterations"]
    args = ["-s", str(dados), "-m", str(saida), "--eval"]
    if cfg.fundo_branco:
        args.append("--white_background")
    for nome, valor in CRONOGRAMA_50K.items():
        args += [f"--{nome}", str(max(1, round(valor * escala)))]
    return args


_RE_METRICAS = re.compile(r"psnr:([\d.]+), ssim:([\d.]+), lpips:([\d.]+), fps:([\d.]+)")
_RE_PROGRESSO = re.compile(r"Training progress:\s+(\d+)%.*?(\d+)/(\d+)")


def ler_metricas(arquivo: Path) -> dict:
    m = _RE_METRICAS.search(Path(arquivo).read_text())
    if not m:
        raise ValueError(f"metric.txt em formato inesperado: {arquivo}")
    psnr, ssim, lpips, fps = map(float, m.groups())
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips, "fps": fps}


def _rodar(cmd: list[str], log: Path, ao_progresso: Callable[[int, int], None] | None) -> None:
    """Executa um script do Ref-Gaussian, gravando a saída em `log` e reportando o passo atual.

    A barra do tqdm vai para o log só a cada mil iterações, para o arquivo não crescer sem necessidade.
    """
    with open(log, "a", encoding="utf-8", buffering=1) as f:  # por linha: dá para acompanhar com tail -f
        proc = subprocess.Popen(cmd, cwd=REFGS_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ultimo = -1
        for linha in _linhas(_pedacos(proc.stdout)):
            m = _RE_PROGRESSO.search(linha)
            if m is None:
                f.write(linha + "\n")
                continue
            passo, total = int(m.group(2)), int(m.group(3))
            if passo // 1000 != ultimo:
                ultimo = passo // 1000
                f.write(linha + "\n")
                if ao_progresso:
                    ao_progresso(passo, total)
        if proc.wait() != 0:
            raise RuntimeError(f"{Path(cmd[1]).name} terminou com código {proc.returncode}; veja {log}")


def _pedacos(fluxo_binario):
    """Lê o que já estiver disponível no pipe (os.read não espera encher um tamanho fixo)."""
    decodificador = codecs.getincrementaldecoder("utf-8")(errors="replace")
    for bloco in iter(lambda: os.read(fluxo_binario.fileno(), 4096), b""):
        yield decodificador.decode(bloco)


def _linhas(pedacos):
    """Quebra a saída em linhas também no retorno de carro, que o tqdm usa para redesenhar a barra."""
    buf = ""
    for pedaco in pedacos:
        buf += pedaco
        *prontas, buf = re.split(r"[\r\n]", buf)
        yield from (p for p in prontas if p.strip())
    if buf.strip():
        yield buf


def treinar(
    dados: Path,
    saida: Path,
    cfg: ConfigRefGS | None = None,
    ao_progresso: Callable[[int, int], None] | None = None,
) -> dict:
    cfg = cfg or ConfigRefGS()
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    log = saida / "refgs.log"

    inicio = time.time()
    _rodar([sys.executable, "train.py", *argumentos_treino(Path(dados), saida, cfg)], log, ao_progresso)
    tempo_treino = time.time() - inicio

    eval_args = ["--model_path", str(saida), "--save_images"]
    if cfg.fundo_branco:
        eval_args.append("--white_background")
    _rodar([sys.executable, "eval.py", *eval_args], log, None)

    metricas = ler_metricas(saida / "metric.txt")
    metricas["tempo_treino_s"] = tempo_treino
    arquivos = {"ply": str(_ultimo_ply(saida)), "log": str(log)}
    for nome in ("env1.png", "env2.png"):
        if (saida / nome).exists():
            arquivos[nome.removesuffix(".png")] = str(saida / nome)
    return {"metricas": metricas, "arquivos": arquivos}


def _ultimo_ply(saida: Path) -> Path:
    pastas = sorted((saida / "point_cloud").glob("iteration_*"), key=lambda p: int(p.name.split("_")[1]))
    if not pastas:
        raise FileNotFoundError(f"nenhum point_cloud salvo em {saida}")
    return pastas[-1] / "point_cloud.ply"
