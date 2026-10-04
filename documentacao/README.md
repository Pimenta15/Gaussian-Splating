# Documentação do projeto e do mestrado

Pasta central para toda a escrita do mestrado (UFPE/CIn) sobre Gaussian Splatting em superfícies reflexivas.

## Estrutura

- `relatorio-ieee/` — relatório incremental no formato IEEE conference
  - `relatorio.tex` — texto principal (pdfLaTeX)
  - `relatorio.pdf` — versão compilada
  - `figs/` — figuras; `figs/make_figs.py <cena> <vista>` regenera os painéis qualitativos a partir de `ref-gaussian/output/<cena>`
  - `template-original-ieee.tex`, `IEEEtran_HOWTO.pdf` — template e manual originais da IEEE, para consulta

## Compilar

Local (MiKTeX): `pdflatex relatorio.tex` duas vezes, dentro de `relatorio-ieee/`.

Overleaf: enviar `relatorio-ieee-overleaf.zip` (New Project → Upload Project) e usar o compilador pdfLaTeX.

## Convenções

- Valores ainda não obtidos aparecem como `\pend` (traço cinza); trechos a completar pelo autor como `\todo{...}` (vermelho).
- Números de outros trabalhos vêm sempre da publicação original, com citação.
