# Código base (não versionado aqui)

A pasta `ref-gaussian/` contém o código oficial do Ref-Gaussian e fica fora deste repositório (`.gitignore`), sem remote configurado.

- Repositório original: https://github.com/fudan-zvg/ref-gaussian
- Commit usado: `af7691aa1097cc6bf89dab6de0fcd358e8830bd8` (2025-03-18)
- Artigo: Yao et al., *Reflective Gaussian Splatting*, ICLR 2025 — https://arxiv.org/abs/2412.19282

Para recriar a pasta:

```
git clone https://github.com/fudan-zvg/ref-gaussian.git
git -C ref-gaussian checkout af7691aa1097cc6bf89dab6de0fcd358e8830bd8
git -C ref-gaussian remote remove origin
```

Depois: `build_submodules.bat` (compila as extensões CUDA) e `run_ref_gaussian.bat <script.py> ...` para treinar/avaliar.
