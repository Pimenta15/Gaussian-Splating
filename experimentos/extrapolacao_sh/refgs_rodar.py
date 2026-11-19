"""Treina o Ref-Gaussian (código oficial, via vitrine_worker.train_refgs) na divisão por elevação e avalia.

    python refgs_rodar.py /dados/banda/car_b25 /exp/saidas/car/refgs_b25 25 30000
"""
import json
import subprocess
import sys
from pathlib import Path

from vitrine_worker.train_refgs import ConfigRefGS, treinar

dados, saida, banda, iteracoes = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
r = treinar(dados, saida, ConfigRefGS(iteracoes=iteracoes))
(saida / "resultado_refgs.json").write_text(json.dumps(r, indent=1))
print("metric.txt (média de todas as vistas de avaliação):", r["metricas"], flush=True)
subprocess.run([sys.executable, str(Path(__file__).parent / "refgs_avaliar.py"), str(dados), str(saida), banda], check=True)
print("ok", flush=True)
