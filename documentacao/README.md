# Documentação do projeto e do mestrado

Pasta central para toda a escrita do mestrado (UFPE/CIn) sobre Gaussian Splatting em superfícies reflexivas.

## Estrutura

- `relatorio-ieee/` — relatório incremental no formato IEEE conference
  - `relatorio.tex` / `relatorio.pdf` — versão enxuta, em linguagem simples (a que está sendo entregue)
  - `relatorio-completo.tex` / `relatorio-completo.pdf` — versão detalhada (equações, decomposição de material, mapa de ambiente, curvas de treino), para consulta e para as próximas entregas
  - `atualizar_resultados.py` — atualiza tabelas e figuras com as cenas concluídas e recompila as duas versões
  - `figs/make_analise.py` — decomposição de material, mapa de ambiente e PSNR por vista (sem GPU)
  - `figs/make_visibilidade.py` — exporta visibilidade, luz direta e indireta por vista e mede o erro nos pixels ocluídos (GPU)
  - `figs/avaliar_checkpoints.py` — avalia checkpoints intermediários com a malha correspondente (GPU)
  - `figs/teste_offset_visibilidade.py` — testa autointerseção deslocando a origem do raio de visibilidade (GPU)
  - Os scripts de GPU rodam a partir de `ref-gaussian` com `run_ref_gaussian.bat ..\documentacao\relatorio-ieee\figs\<script>.py -m output/<cena> --white_background`
  - `figs/` — figuras; `figs/make_figs.py <cena> <vista>` regenera os painéis qualitativos a partir de `ref-gaussian/output/<cena>`
  - `template-original-ieee.tex`, `IEEEtran_HOWTO.pdf` — template e manual originais da IEEE, para consulta

## Compilar

Local (MiKTeX): `pdflatex relatorio.tex` duas vezes, dentro de `relatorio-ieee/`.

Overleaf: enviar `relatorio-ieee-overleaf.zip` (New Project → Upload Project) e usar o compilador pdfLaTeX.

## Convenções

- Valores ainda não obtidos aparecem como `\pend` (traço cinza); trechos a completar pelo autor como `\todo{...}` (vermelho).
- Números de outros trabalhos vêm sempre da publicação original, com citação.
