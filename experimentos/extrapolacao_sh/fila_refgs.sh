#!/bin/bash
# Fila do Ref-Gaussian: renders dos modelos completos para a demo, oráculos, depois treino + oráculos nas outras cenas.
EXP="C:\Users\Pimenta\Desktop\Gaussian-Splating\experimentos\extrapolacao_sh"
DADOS="C:\Users\Pimenta\Desktop\Gaussian-Splating\ref-gaussian\data\ref_nerf"
REFOUT="C:\Users\Pimenta\Desktop\Gaussian-Splating\ref-gaussian\output"
FONTE="C:\Users\Pimenta\Desktop\vitrine3d\services\worker\vitrine_worker"
PY="C:/Users/Pimenta/AppData/Local/Programs/Python/Python311/python.exe"
LOG=saidas/fila_refgs.log
roda() {
  MSYS_NO_PATHCONV=1 docker run --rm --gpus all -v "$DADOS:/data:ro" -v "$DADOS:/dados/ref_nerf:ro" \
    -v "$DADOS:/opt/ref-gaussian/data/ref_nerf:ro" -v "$REFOUT:/opt/ref-gaussian/output:ro" \
    -v "$EXP\dados_banda:/dados/banda" -v "$EXP:/exp" -v "$FONTE:/app/vitrine_worker:ro" \
    -v vitrine3d-cache:/cache -e TORCH_HOME=/cache/torch -w /exp vitrine3d-worker:dev "$@"
}
oraculo() {
  echo "$(date +%H:%M) oraculo $1" >> $LOG
  roda python /exp/refgs_oraculo.py -m /exp/saidas/$1/refgs_b25 --passos 1000 > saidas/$1/refgs_b25/oraculo.log 2>&1
  grep -E "^(original|material|ambiente|indireta|aparencia) |Error" saidas/$1/refgs_b25/oraculo.log | cut -c1-160 >> $LOG
}
for c in car toaster helmet coffee teapot ball; do
  [ -f saidas/$c/b25_sh3/explorar/modelos.json ] && continue
  echo "$(date +%H:%M) demo_modelos $c" >> $LOG
  roda python /exp/demo_modelos.py --cena $c > saidas/$c/b25_sh3/demo_modelos.log 2>&1
  tail -1 saidas/$c/b25_sh3/demo_modelos.log >> $LOG
done
for c in car helmet toaster; do oraculo $c; done
for c in coffee teapot ball; do
  p=saidas/$c/refgs_b25; mkdir -p $p
  if [ ! -f $p/por_vista.json ]; then
    echo "$(date +%H:%M) treino refgs $c" >> $LOG
    roda python /exp/refgs_rodar.py /dados/banda/${c}_b25 /exp/saidas/$c/refgs_b25 25 28000 > $p/run.log 2>&1
    grep -E "^(faixa|acima|cima):" $p/run.log >> $LOG
  fi
  $PY refgs_geometria.py dados_banda/${c}_b25 ../../ref-gaussian/data/ref_nerf $c $p 25 2>&1 | grep refgs >> $LOG
done
for c in coffee teapot ball; do oraculo $c; done
echo "$(date +%H:%M) fim" >> $LOG
