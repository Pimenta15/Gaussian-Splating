"""Monta, para o Ref-Gaussian, a mesma divisão por elevação usada no 3DGS (rodar.py).

dados_banda/<cena>_b<banda>/transforms_train.json : só as vistas de treino abaixo da banda
dados_banda/<cena>_b<banda>/transforms_test.json  : as 200 de teste + as de treino não usadas
Os caminhos apontam para ../../ref_nerf/<cena>/..., então no contêiner o original é montado em
/dados/ref_nerf e esta pasta em /dados/banda.

    python prepara_refgs.py <ref_nerf> <dados_banda> car 25
"""
import json
import math
import sys
from pathlib import Path


def elev(fr):
    t = [r[3] for r in fr["transform_matrix"][:3]]
    return math.degrees(math.asin(t[2] / math.sqrt(sum(x * x for x in t))))


def main():
    ref, destino, cena, banda = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], float(sys.argv[4])
    tr = json.loads((ref / cena / "transforms_train.json").read_text())
    te = json.loads((ref / cena / "transforms_test.json").read_text())
    pasta = destino / f"{cena}_b{banda:g}"
    pasta.mkdir(parents=True, exist_ok=True)

    def caminho(fr, sub):
        nome = Path(fr["file_path"]).name
        return f"../../ref_nerf/{cena}/{sub}/{nome}"

    treino = [dict(fr, file_path=caminho(fr, "train")) for fr in tr["frames"] if 0 <= elev(fr) < banda]
    teste = [dict(fr, file_path=caminho(fr, "test"), origem="teste") for fr in te["frames"]]
    teste += [dict(fr, file_path=caminho(fr, "train"), origem="treino_nao_usada") for fr in tr["frames"]
              if elev(fr) >= banda]
    (pasta / "transforms_train.json").write_text(json.dumps({"camera_angle_x": tr["camera_angle_x"], "frames": treino}))
    (pasta / "transforms_test.json").write_text(json.dumps({"camera_angle_x": te["camera_angle_x"], "frames": teste}))
    print(pasta, len(treino), "treino", len(teste), "avaliação")


if __name__ == "__main__":
    main()
