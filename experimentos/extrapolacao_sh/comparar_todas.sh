#!/bin/bash
# Roda comparar.py nas seis cenas, uma por vez na GPU.
EXP="C:\Users\Pimenta\Desktop\Gaussian-Splating\experimentos\extrapolacao_sh"
DADOS="C:\Users\Pimenta\Desktop\Gaussian-Splating\ref-gaussian\data\ref_nerf"
FONTE="C:\Users\Pimenta\Desktop\vitrine3d\services\worker\vitrine_worker"
for c in car toaster helmet coffee teapot ball; do
  [ -f "saidas/$c/b25_sh3/comparacao.json" ] && continue
  echo "$(date +%H:%M) $c" >> saidas/comparar.log
  MSYS_NO_PATHCONV=1 docker run --rm --gpus all -v "$DADOS:/data:ro" -v "$EXP:/exp" -v "$FONTE:/app/vitrine_worker:ro" \
    -v vitrine3d-cache:/cache -e TORCH_HOME=/cache/torch -w /exp vitrine3d-worker:dev \
    python /exp/comparar.py --cena $c --banda 25 > "saidas/$c/b25_sh3/comparar.log" 2>&1
  tail -1 "saidas/$c/b25_sh3/comparar.log" >> saidas/comparar.log
done
echo "$(date +%H:%M) fim" >> saidas/comparar.log
