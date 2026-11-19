# Extrapolação em elevação no 3D Gaussian Splatting

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

## Reproduzir os experimentos

Os experimentos rodam no contêiner `vitrine3d-worker:dev` (CUDA 11.8, gsplat 1.5.3, PyTorch). Os dados são o Shiny Blender (Ref-NeRF), em `ref_nerf/<cena>`.

| Script | O que faz |
| --- | --- |
| `rodar.py` | Treina o 3DGS só com as vistas abaixo da faixa (`--banda 25`) e avalia o original, só DC, projeção global e projeção por visibilidade em todos os grupos de elevação |
| `comparar.py` | Compara, no mesmo modelo, as correções leves: truncamento de grau, grau adaptativo por Gaussiana, projeções e projeção + grau adaptativo. Mede qualidade, tempo e armazenamento e salva a varredura de elevação da demonstração |
| `oraculo.py`, `oraculo2.py` | Oráculos: reajustam só a cor, só a cor constante ou só a geometria com vistas de cima. Ajustam nas vistas pares e avaliam nas ímpares, ou ajustam num lado do objeto e avaliam no outro |
| `geometria.py` | IoU da silhueta e erro angular das normais, pela profundidade e pelo eixo mais curto, contra a referência do Shiny Blender |
| `refgs_rodar.py`, `refgs_avaliar.py`, `refgs_geometria.py` | Treina e avalia o Ref-Gaussian nas mesmas vistas e mede as normais dele |
| `cobertura_reflexo.py` | Testa se as direções refletidas vistas de cima já aparecem no treino, usando as normais de referência |
| `carro_real.py` | Mede, num veículo real sem vista de cima, quanto a projeção muda a imagem em cada elevação |
| `tabelas.py` | Gera todas as tabelas e figuras do artigo a partir dos JSON |
| `fila.ps1`, `comparar_todas.sh` | Filas que rodam tudo, uma execução por vez na GPU |

Hardware usado: notebook com RTX 4050 de 6 GB.
