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
             "refgs_completo", "holdout"]


def refgs_holdout(saidas: Path, cena: str, modelo: str = "refgs_holdout", divisao: str = "") -> dict:
    """(origem na varredura, nome) -> (imagem em data URI, PSNR) de um Ref-Gaussian, a partir das renderizações que o
    eval.py salva: o treinado sem as fotos avaliadas (padrão) ou o treinado só na faixa (modelo='refgs_b25')."""
    from io import BytesIO  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415
    pasta = saidas / cena / modelo
    f_tr = (saidas.parent / divisao / "transforms_test.json" if divisao
            else saidas.parent / "dados_holdout" / cena / "transforms_test.json")
    if not (pasta / "por_vista.json").exists() or not f_tr.exists():
        return {}
    psnr = {(ln["origem"], ln["nome"]): ln["refgs"]["psnr"] for ln in json.loads((pasta / "por_vista.json").read_text())}
    out = {}
    for i, fr in enumerate(json.loads(f_tr.read_text())["frames"]):
        orig, nome = fr["origem"], Path(fr["file_path"]).name
        chave = ("teste" if orig == "teste" else "treino_nao_usada", nome)
        img = pasta / "test" / "renders" / "rgb" / f"{i:05d}.png"
        if img.exists() and (orig, nome) in psnr:
            buf = BytesIO()
            Image.open(img).convert("RGB").resize((400, 400), Image.LANCZOS).save(buf, "JPEG", quality=88)
            out[chave] = ("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), psnr[(orig, nome)])
    return out


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
        rg_hold = refgs_holdout(saidas, c)
        rg_faixa = refgs_holdout(saidas, c, "refgs_b25", f"dados_banda/{c}_b25")
        vistas = []
        for m in meta:
            cand = [ln for ln in rg if ln["origem"] == m["origem"] and abs(ln["elev"] - m["elev"]) < 0.01
                    and abs(az.get((ln["origem"], ln["nome"]), 1e9) - m["azim"]) < 0.5]
            if len(cand) == 1:
                m["psnr"]["refgs"] = cand[0]["refgs"]["psnr"]
            mod = modelos.get(m["i"], {})
            for k_src, k_dst in (("psnr_completo", "completo"), ("psnr_refgs_completo", "refgs_completo"),
                                 ("psnr_holdout", "holdout")):
                if k_src in mod:
                    m["psnr"][k_dst] = mod[k_src]
            m["nome"] = mod.get("nome")
            imgs = {}
            for v in VARIANTES:
                f = pasta / f"{m['i']:02d}_{v}.jpg"
                if f.exists():
                    imgs[v] = "data:image/jpeg;base64," + base64.b64encode(f.read_bytes()).decode()
            rh = rg_hold.get((m["origem"], m["nome"]))
            if rh:
                imgs["refgs_holdout"], m["psnr"]["refgs_holdout"] = rh
            if "refgs" not in imgs:  # Ref-Gaussian da faixa treinado depois da varredura: usa as renderizações dele
                rb = rg_faixa.get((m["origem"], m["nome"]))
                if rb:
                    imgs["refgs"], m["psnr"]["refgs"] = rb
            vistas.append({"elev": round(m["elev"], 1), "azim": round(m["azim"], 1), "origem": m["origem"],
                           "treino_completos": m["origem"] != "teste",
                           "psnr": {k: round(v, 2) for k, v in m["psnr"].items()}, "img": imgs})
        res = json.loads((saidas / c / "b25_sh3" / "resultado.json").read_text())
        f_h = saidas / c / "holdout_sh3" / "resultado.json"
        hold = json.loads(f_h.read_text()) if f_h.exists() else None
        dados = {"cena": c, "n_gaussianas": comp["n_gaussianas"], "n_treino": res["n_treino"],
                 "holdout": {"n_treino": hold["n_treino"], "elev_max": round(max(hold["elev_treino"]))} if hold else None,
                 "mb": {k: round(v, 1) for k, v in comp["mb"].items()},
                 "tempo_s": {k: round(v, 3) for k, v in comp["tempo_s"].items()}, "vistas": vistas}
        (destino / "cenas" / f"{c}.json").write_text(json.dumps(dados, separators=(",", ":")))
        indice.append({"cena": c, "tem_refgs": any("refgs" in v["img"] for v in vistas)})
        print(c, len(vistas), "vistas", round((destino / "cenas" / f"{c}.json").stat().st_size / 2 ** 20, 1), "MB")
    (destino / "cenas" / "indice.json").write_text(json.dumps(indice))


if __name__ == "__main__":
    main()
