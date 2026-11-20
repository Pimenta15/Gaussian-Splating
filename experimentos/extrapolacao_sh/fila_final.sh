#!/bin/bash
# Fila final: modelos com todas as alturas sem as fotos de avaliação, depois o restante do Ref-Gaussian.
EXP="C:\Users\Pimenta\Desktop\Gaussian-Splating\experimentos\extrapolacao_sh"
DADOS="C:\Users\Pimenta\Desktop\Gaussian-Splating\ref-gaussian\data\ref_nerf"
REFOUT="C:\Users\Pimenta\Desktop\Gaussian-Splating\ref-gaussian\output"
FONTE="C:\Users\Pimenta\Desktop\vitrine3d\services\worker\vitrine_worker"
PY="C:/Users/Pimenta/AppData/Local/Programs/Python/Python311/python.exe"
LOG=saidas/fila_final.log
roda() {
  MSYS_NO_PATHCONV=1 docker run --rm --gpus all -v "$DADOS:/data:ro" -v "$DADOS:/dados/ref_nerf:ro" \
    -v "$DADOS:/opt/ref-gaussian/data/ref_nerf:ro" -v "$REFOUT:/opt/ref-gaussian/output:ro" \
    -v "$EXP\dados_banda:/dados/banda" -v "$EXP\dados_holdout:/dados/holdout" -v "$EXP:/exp" \
    -v "$FONTE:/app/vitrine_worker:ro" -v vitrine3d-cache:/cache -e TORCH_HOME=/cache/torch -w /exp \
    vitrine3d-worker:dev "$@"
}
# espera o Ref-Gaussian do coffee terminar e para as filas antigas
until [ -f saidas/coffee/refgs_b25/por_vista.json ]; do sleep 15; done
sleep 3
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*fila_refgs*' -or \$_.CommandLine -like '*fila_geo*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
for i in $(docker ps -q --filter ancestor=vitrine3d-worker:dev); do docker stop $i > /dev/null; done
echo "$(date +%H:%M) filas antigas paradas" >> $LOG
geometria_refgs() {
  p=saidas/$1/refgs_b25
  [ -f $p/geometria.json ] || $PY refgs_geometria.py dados_banda/${1}_b25 ../../ref-gaussian/data/ref_nerf $1 $p 25 2>&1 | grep refgs >> $LOG
}
oraculo() {
  echo "$(date +%H:%M) oraculo $1" >> $LOG
  roda python /exp/refgs_oraculo.py -m /exp/saidas/$1/refgs_b25 --passos 1000 > saidas/$1/refgs_b25/oraculo_$2.log 2>&1
  grep -E "^(original|material|ambiente|indireta|aparencia|geometria|tudo) |Error" saidas/$1/refgs_b25/oraculo_$2.log | cut -c1-140 >> $LOG
}
geometria_refgs coffee
# 1) 3DGS com todas as alturas, sem as fotos de avaliação
for c in car helmet toaster coffee teapot ball; do
  [ -f saidas/$c/holdout_sh3/resultado.json ] && continue
  mkdir -p saidas/$c/holdout_sh3
  echo "$(date +%H:%M) 3dgs holdout $c" >> $LOG
  roda python /exp/rodar_holdout.py --cena $c > saidas/$c/holdout_sh3/run.log 2>&1
  grep -E "separadas de cima" saidas/$c/holdout_sh3/run.log >> $LOG
done
# 2) Ref-Gaussian com todas as alturas, sem as fotos de avaliação
for c in car helmet toaster; do
  p=saidas/$c/refgs_holdout; mkdir -p $p
  [ -f $p/por_vista.json ] && continue
  echo "$(date +%H:%M) refgs holdout $c" >> $LOG
  roda python /exp/refgs_rodar.py /dados/holdout/$c /exp/saidas/$c/refgs_holdout 25 28000 > $p/run.log 2>&1
  grep -E "^(faixa|acima|cima):" $p/run.log >> $LOG
done
# 3) restante do Ref-Gaussian na faixa
for c in teapot ball; do
  p=saidas/$c/refgs_b25; mkdir -p $p
  if [ ! -f $p/por_vista.json ]; then
    echo "$(date +%H:%M) treino refgs $c" >> $LOG
    roda python /exp/refgs_rodar.py /dados/banda/${c}_b25 /exp/saidas/$c/refgs_b25 25 28000 > $p/run.log 2>&1
    grep -E "^(faixa|acima|cima):" $p/run.log >> $LOG
  fi
  geometria_refgs $c
done
for c in coffee teapot ball car helmet toaster; do oraculo $c final; done
echo "$(date +%H:%M) fim" >> $LOG
