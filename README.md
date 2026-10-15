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

## Resultados (Shiny Blender, 6 cenas)

Modelo final (50 mil iterações), avaliado com o `eval.py` oficial. Entre parênteses, o valor do artigo.

| Cena | PSNR | SSIM | LPIPS |
|---|---|---|---|
| ball | 36,07 (37,01) | 0,985 (0,981) | 0,089 (0,098) |
| car | 31,25 (31,04) | 0,966 (0,964) | 0,031 (0,033) |
| coffee | 34,29 (34,63) | 0,976 (0,976) | 0,077 (0,076) |
| helmet | 30,45 (32,32) | 0,965 (0,971) | 0,061 (0,049) |
| teapot | 47,19 (47,16) | 0,998 (0,998) | 0,006 (0,006) |
| toaster | 27,51 (28,05) | 0,948 (0,948) | 0,074 (0,074) |
| **média** | **34,46 (35,04)** | **0,973 (0,973)** | **0,057 (0,056)** |

Com o checkpoint de 40 mil iterações e a malha correspondente, toaster chega a 28,08 / 0,951 / 0,068 e helmet a 32,09 / 0,971 / 0,050. Detalhes, análise de visibilidade e discussão estão no relatório (`documentacao/relatorio-ieee`).
