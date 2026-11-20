"""Divisão "todas as alturas, sem as fotos de avaliação".

Das 100 vistas de treino, separa as que servem de avaliação acima da faixa:
  - as vistas de cima (>= 50°) de índice ímpar, as mesmas avaliadas pelos oráculos;
  - as vistas de treino que aparecem na varredura de elevação da demonstração.
O modelo é treinado com as demais (todas as alturas continuam cobertas) e avaliado nas separadas, que nunca viu.

    python prepara_holdout.py <ref_nerf> <dados_holdout> <saidas> car
Gera <dados_holdout>/<cena>/transforms_train.json e transforms_test.json (teste + separadas, com campo origem) e
<saidas>/<cena>/holdout.json com os nomes separados.
"""
import json
import math
import sys
from pathlib import Path


def elev(fr):
    t = [r[3] for r in fr["transform_matrix"][:3]]
    return math.degrees(math.asin(t[2] / math.sqrt(sum(x * x for x in t))))


def main():
    ref, destino, saidas, cena = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    tr = json.loads((ref / cena / "transforms_train.json").read_text())
    te = json.loads((ref / cena / "transforms_test.json").read_text())
    nome = lambda fr: Path(fr["file_path"]).name  # noqa: E731
    # 1) as vistas de treino que a varredura da demonstração mostra precisam ficar fora do treino
    separadas = set()
    f_mod = saidas / cena / "b25_sh3" / "explorar" / "modelos.json"
    if f_mod.exists():
        separadas |= {m["nome"] for m in json.loads(f_mod.read_text()) if m.get("treino_dos_completos") and m["nome"]}
    # 2) das demais vistas de cima, ordenadas pela elevação, uma sim, uma não, mantendo a mais alta no treino,
    #    para que o treino continue cobrindo todas as alturas
    cand = sorted((fr for fr in tr["frames"] if elev(fr) >= 50 and nome(fr) not in separadas), key=elev)
    alternadas = cand[1::2] if (len(cand) - 1) % 2 == 0 else cand[0::2]
    separadas |= {nome(fr) for fr in alternadas}

    def caminho(fr, pasta):
        return f"../../ref_nerf/{cena}/{pasta}/{nome(fr)}"

    pasta = destino / cena
    pasta.mkdir(parents=True, exist_ok=True)
    treino = [dict(fr, file_path=caminho(fr, "train")) for fr in tr["frames"] if nome(fr) not in separadas]
    teste = [dict(fr, file_path=caminho(fr, "test"), origem="teste") for fr in te["frames"]]
    teste += [dict(fr, file_path=caminho(fr, "train"), origem="treino_separada") for fr in tr["frames"]
              if nome(fr) in separadas]
    base = {k: v for k, v in tr.items() if k != "frames"}
    (pasta / "transforms_train.json").write_text(json.dumps(dict(base, frames=treino), indent=1))
    (pasta / "transforms_test.json").write_text(json.dumps(dict(base, frames=teste), indent=1))
    (saidas / cena / "holdout.json").write_text(json.dumps(sorted(separadas), indent=1))
    el = [elev(fr) for fr in treino]
    print(f"{cena}: {len(treino)} vistas de treino ({min(el):.0f}° a {max(el):.0f}°), {len(separadas)} separadas")


if __name__ == "__main__":
    main()
