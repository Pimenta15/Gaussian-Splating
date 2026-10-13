# Gaussian-Splating

Repositório do meu mestrado (UFPE/CIn) sobre reconstrução 3D com Gaussian Splatting em superfícies reflexivas.

O ponto de partida é o **Ref-Gaussian** (Yao et al., ICLR 2025), cujo código oficial uso como baseline: https://github.com/fudan-zvg/ref-gaussian. O código original fica fora deste repositório (ver [UPSTREAM.md](UPSTREAM.md)).

## Conteúdo

- `documentacao/` — relatório no formato IEEE (versão enxuta e versão detalhada), figuras e script que atualiza as tabelas a partir dos treinos. Ver [documentacao/README.md](documentacao/README.md).
- `build_submodules.bat` — compila as extensões CUDA do Ref-Gaussian no Windows (CUDA 11.8 + Visual Studio 2022).
- `run_ref_gaussian.bat` — roda `train.py` / `eval.py` no ambiente do compilador.
- `run_queue_shiny.bat` — treina e avalia em fila as cenas do Shiny Blender.
- `watch_train.ps1` — acompanha o progresso do treino pelo terminal.

## Como reproduzir

1. Clonar o código base e criar o ambiente conda `ref-gaussian` (Python 3.8, PyTorch 2.0.0 + CUDA 11.8), como descrito em [UPSTREAM.md](UPSTREAM.md).
2. Compilar as extensões: `build_submodules.bat`.
3. Treinar uma cena: `run_ref_gaussian.bat train.py -s data/ref_nerf/toaster -m output/toaster --eval --white_background`.
4. Avaliar: `run_ref_gaussian.bat eval.py --white_background --save_images --model_path output/toaster`.

## Resultados até agora (Shiny Blender)

| Cena | PSNR | SSIM | LPIPS |
|---|---|---|---|
| toaster | 27,51 | 0,948 | 0,074 |
| ball | 36,07 | 0,985 | 0,089 |

As demais cenas estão em andamento. Os detalhes e a comparação com o artigo estão no relatório.
