"""Empacota as imagens da varredura de elevação (comparar.py) em um JSON por cena para a demonstração web.

    python empacotar_demo.py saidas ../../documentacao/demo
"""
import base64
import json
import math
import sys
from pathlib import Path

CENAS = ["car", "toaster", "helmet", "coffee", "teapot", "ball"]
VARIANTES = ["gt", "original", "grau0", "grau1", "global", "global_adapt", "visib", "visib_adapt", "completo", "refgs",
             "refgs_completo"]


def main():
    saidas, destino = Path(sys.argv[1]), Path(sys.argv[2])
    (destino / "cenas").mkdir(parents=True, exist_ok=True)
    indice = []
    for c in CENAS:
        pasta = saidas / c / "b25_sh3" / "explorar"
        if not (pasta / "meta.json").exists():
            continue
        meta = json.loads((pasta / "meta.json").read_text())
        comp = json.loads((saidas / c / "b25_sh3" / "comparacao.json").read_text())
        f_rg = saidas / c / "refgs_b25" / "por_vista.json"
        rg = json.loads(f_rg.read_text()) if f_rg.exists() else []
        # azimute de cada vista do Ref-Gaussian, pelo mesmo cálculo de rodar.azimute (as vistas de teste do Shiny
        # Blender vêm em pares com a mesma elevação, então a elevação sozinha não identifica a vista)
        f_tr = saidas.parent / "dados_banda" / f"{c}_b25" / "transforms_test.json"
        az = {}
        if f_tr.exists():
            for fr in json.loads(f_tr.read_text())["frames"]:
                t = [r[3] for r in fr["transform_matrix"][:3]]
                az[(fr["origem"], Path(fr["file_path"]).name)] = math.degrees(math.atan2(t[1], t[0]))
        f_mod = pasta / "modelos.json"
        modelos = {d["i"]: d for d in json.loads(f_mod.read_text())} if f_mod.exists() else {}
        vistas = []
        for m in meta:
            cand = [ln for ln in rg if ln["origem"] == m["origem"] and abs(ln["elev"] - m["elev"]) < 0.01
                    and abs(az.get((ln["origem"], ln["nome"]), 1e9) - m["azim"]) < 0.5]
            if len(cand) == 1:
                m["psnr"]["refgs"] = cand[0]["refgs"]["psnr"]
            mod = modelos.get(m["i"], {})
            for k_src, k_dst in (("psnr_completo", "completo"), ("psnr_refgs_completo", "refgs_completo")):
                if k_src in mod:
                    m["psnr"][k_dst] = mod[k_src]
            imgs = {}
            for v in VARIANTES:
                f = pasta / f"{m['i']:02d}_{v}.jpg"
                if f.exists():
                    imgs[v] = "data:image/jpeg;base64," + base64.b64encode(f.read_bytes()).decode()
            vistas.append({"elev": round(m["elev"], 1), "azim": round(m["azim"], 1), "origem": m["origem"],
                           "treino_completos": m["origem"] != "teste",
                           "psnr": {k: round(v, 2) for k, v in m["psnr"].items()}, "img": imgs})
        res = json.loads((saidas / c / "b25_sh3" / "resultado.json").read_text())
        dados = {"cena": c, "n_gaussianas": comp["n_gaussianas"], "n_treino": res["n_treino"],
                 "mb": {k: round(v, 1) for k, v in comp["mb"].items()},
                 "tempo_s": {k: round(v, 3) for k, v in comp["tempo_s"].items()}, "vistas": vistas}
        (destino / "cenas" / f"{c}.json").write_text(json.dumps(dados, separators=(",", ":")))
        indice.append({"cena": c, "tem_refgs": any("refgs" in v["img"] for v in vistas)})
        print(c, len(vistas), "vistas", round((destino / "cenas" / f"{c}.json").stat().st_size / 2 ** 20, 1), "MB")
    (destino / "cenas" / "indice.json").write_text(json.dumps(indice))


if __name__ == "__main__":
    main()
