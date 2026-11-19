#!/bin/bash
# Depois da fila do Ref-Gaussian: oráculos de geometria e "tudo" nas três primeiras cenas.
EXP="C:\Users\Pimenta\Desktop\Gaussian-Splating\experimentos\extrapolacao_sh"
DADOS="C:\Users\Pimenta\Desktop\Gaussian-Splating\ref-gaussian\data\ref_nerf"
FONTE="C:\Users\Pimenta\Desktop\vitrine3d\services\worker\vitrine_worker"
LOG=saidas/fila_refgs.log
until grep -q " fim" $LOG 2>/dev/null; do sleep 30; done
for c in car helmet toaster; do
  echo "$(date +%H:%M) oraculo geometria $c" >> $LOG
  MSYS_NO_PATHCONV=1 docker run --rm --gpus all -v "$DADOS:/data:ro" -v "$DADOS:/dados/ref_nerf:ro" \
    -v "$EXP\dados_banda:/dados/banda" -v "$EXP:/exp" -v "$FONTE:/app/vitrine_worker:ro" \
    -v vitrine3d-cache:/cache -e TORCH_HOME=/cache/torch -w /exp vitrine3d-worker:dev \
    python /exp/refgs_oraculo.py -m /exp/saidas/$c/refgs_b25 --passos 1000 > saidas/$c/refgs_b25/oraculo_geo.log 2>&1
  grep -E "^(geometria|tudo) |Error" saidas/$c/refgs_b25/oraculo_geo.log | cut -c1-160 >> $LOG
done
echo "$(date +%H:%M) fim geo" >> $LOG
