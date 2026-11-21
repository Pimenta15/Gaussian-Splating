# Extrapolação em elevação no 3DGS e no Ref-Gaussian

Código do artigo *Reflexos Fora do Ângulo Filmado: 3DGS e Ref-Gaussian sob Cobertura Limitada de Elevação na Reconstrução de Veículos*.
Todos os números e tabelas do artigo são gerados por estes scripts a partir dos resultados por vista (JSON).
A demonstração interativa está em `documentacao/demo` (abra `index.html` por um servidor local, por exemplo `python -m http.server`).

## Uso rápido: corrigir um modelo `.ply` (CPU, sem treino)

```bash
python projetar_ply.py modelo.ply modelo_proj.ply --elev 0.4 3.8 --relatorio
```

- `--elev MIN MAX`: elevações (graus) das câmeras que filmaram o objeto, eixo z para cima. Também é possível ler as elevações do `cameras.json` do 3DGS original com `--cameras`.
- Lê e grava o `.ply` no layout do 3DGS original e precisa só de `numpy`. Leva cerca de 0,6 s para 273 mil Gaussianas.
- `--relatorio` imprime o espectro da matriz de Gram da faixa filmada e a energia direcional removida.
- O resultado abre em qualquer visualizador de 3DGS que leia `.ply` com harmônicos esféricos, como o SuperSplat no navegador. Abra os dois arquivos e compare a mesma vista vista de cima.

## Ambiente

Os experimentos rodam num contêiner com CUDA 11.8, PyTorch 2.0, gsplat 1.5.3 e o código oficial do Ref-Gaussian (commit `af7691a`), descrito no `Dockerfile` desta pasta:

```bash
docker build -t vitrine3d-worker:dev .
```

O pacote `vitrine_worker/` traz o treino do 3DGS (`train_3dgs.py`), a chamada ao Ref-Gaussian oficial (`train_refgs.py`, que roda o `train.py` e o `eval.py` dele em `REFGS_DIR`), a leitura do formato Blender (`datasets.py`), as métricas (`metrics.py`) e a projeção dos harmônicos esféricos (`sh_observado.py`). Como os scripts rodam com `-w /exp`, o pacote é importado direto desta pasta.

Os dados são o Shiny Blender, do Ref-NeRF ([Verbin et al., 2022](https://dorverbin.github.io/refnerf/)), com uma pasta por cena em `ref_nerf/<cena>` (`car`, `toaster`, `helmet`, `coffee`, `teapot`, `ball`). Exemplo de execução:

```bash
docker run --rm --gpus all -v "$PWD/../../ref-gaussian/data/ref_nerf:/data:ro" -v "$PWD:/exp" \
  -v vitrine3d-cache:/cache -e TORCH_HOME=/cache/torch -w /exp vitrine3d-worker:dev \
  python rodar.py --dados /data --cena car --banda 25
```

As filas (`fila*.sh`, `fila.ps1`, `comparar_todas.sh`) usam os caminhos da máquina em que os experimentos rodaram: troque `EXP` e `DADOS` no início de cada uma.

## Scripts

| Script | O que faz |
| --- | --- |
| `rodar.py` | Treina o 3DGS só com as vistas abaixo da faixa (`--banda 25`) e avalia o original, só DC, projeção global e projeção por visibilidade em todos os grupos de elevação |
| `comparar.py` | Compara, no mesmo modelo, as correções leves: truncamento de grau, grau adaptativo por Gaussiana, projeções e projeção + grau adaptativo. Mede qualidade, tempo e armazenamento e salva a varredura de elevação da demonstração |
| `oraculo.py`, `oraculo2.py` | Oráculos do 3DGS: reajustam só a cor, só a cor constante ou só a geometria com vistas de cima. Ajustam nas vistas pares e avaliam nas ímpares, ou ajustam num lado do objeto e avaliam no outro |
| `geometria.py` | IoU da silhueta e erro angular das normais, pela profundidade e pelo eixo mais curto, contra a referência do Shiny Blender |
| `prepara_refgs.py` | Monta para o Ref-Gaussian a mesma divisão por elevação do 3DGS (`dados_banda/`) |
| `refgs_rodar.py`, `refgs_avaliar.py`, `refgs_geometria.py` | Treina e avalia o Ref-Gaussian nas mesmas vistas (28 mil iterações, cronograma escalado) e mede as normais renderizadas por ele |
| `refgs_oraculo.py` | Oráculos do Ref-Gaussian: reajustam com vistas de cima só o material, só o ambiente, só a luz indireta, toda a aparência, só a geometria ou tudo |
| `refgs_iteracoes.py` | Controle por *checkpoints*: avalia o Ref-Gaussian completo com 30, 40 e 50 mil iterações |
| `cobertura_reflexo.py` | Testa se as direções refletidas vistas de cima já aparecem no treino, usando as normais de referência |
| `prepara_holdout.py`, `rodar_holdout.py` | Divisão "todas as alturas, sem as fotos avaliadas" e o 3DGS treinado nela (o Ref-Gaussian usa `refgs_rodar.py` na mesma divisão) |
| `resumo_refgs.py` | Resumo 3DGS × Ref-Gaussian por grupo de elevação |
| `demo_modelos.py`, `demo_holdout.py`, `empacotar_demo.py` | Renderizam os modelos de referência nas vistas da varredura e empacotam as imagens da demonstração |
| `tabelas.py` | Gera todas as tabelas e figuras do artigo a partir dos JSON |
| `espectro.py`, `fig_protocolo.py`, `fig_teaser.py`, `fig_slides.py` | Figuras do artigo e dos slides |
| `carro_real.py` | Mede, num veículo real sem vista de cima, quanto a projeção muda a imagem em cada elevação. Depende do pipeline de reconstrução de vídeo e de dados próprios, que não estão neste repositório |

Hardware usado: notebook com RTX 4050 de 6 GB. Tempos de treino por cena na Tabela de modelos do artigo (3DGS: 20 min; Ref-Gaussian na faixa: 47 min).
