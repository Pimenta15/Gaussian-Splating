"""Agrega saidas/<cena>/<rotulo>/resultado.json em tabelas LaTeX e figuras do artigo.

    python tabelas.py <pasta_saidas> <pasta_artigo>

Grupos de vistas (todas inéditas para o modelo treinado na faixa):
  faixa : vistas de TESTE abaixo da elevação máxima de treino (interpolação)
  acima : vistas de TESTE entre a faixa e 47° (extrapolação moderada; o modelo com cobertura completa
          também é avaliado nelas, então dá para medir a lacuna)
  cima  : vistas de TREINO não usadas, acima de 50° (até 89°, praticamente vista de topo)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CENAS = ["ball", "car", "coffee", "helmet", "teapot", "toaster"]
NOME_VAR = {"original": "3DGS (original)", "dc": "Só termo constante (DC)", "global": "Projeção global",
            "visib": "Projeção por visibilidade"}
GRUPOS = ["faixa", "acima", "cima"]


def carregar(saidas: Path, rotulo: str) -> dict:
    out = {}
    for c in CENAS:
        f = saidas / c / rotulo / "resultado.json"
        if f.exists():
            out[c] = json.loads(f.read_text())
    return out


def grupo(linha: dict, banda: float) -> str | None:
    e, origem = linha["elev"], linha["origem"]
    if origem == "teste":
        return "faixa" if e < banda else "acima"
    return "cima" if e >= 50 else None


def medias(res: dict, banda: float, variantes: list[str]) -> dict:
    """{grupo: {variante: {metrica: média por cena}}} e o mesmo por cena."""
    por_cena = {}
    for c, r in res.items():
        por_cena[c] = {}
        for g in GRUPOS:
            sel = [ln for ln in r["por_vista"] if grupo(ln, banda) == g]
            if not sel:
                continue
            por_cena[c][g] = {v: {m: float(np.mean([ln[v][m] for ln in sel])) for m in ("psnr", "ssim", "lpips")}
                              for v in variantes if v in sel[0]}
            por_cena[c][g]["n"] = len(sel)
    return por_cena


def media_cenas(por_cena: dict, g: str, v: str, m: str) -> float | None:
    xs = [pc[g][v][m] for pc in por_cena.values() if g in pc and v in pc[g]]
    return float(np.mean(xs)) if xs else None


def fmt(x, casas=2):
    return "--" if x is None else f"{x:.{casas}f}".replace(".", "{,}")


def tabela_principal(pc: dict, pc_ref: dict, banda: float, n_cenas: int) -> str:
    var = ["original", "dc", "global", "visib"]
    linhas = [r"\begin{table*}[t]", r"\centering",
              rf"\caption{{Média nas {n_cenas} cenas do Shiny Blender, treino só com vistas abaixo de {banda:g}°. "
              r"Faixa e acima: vistas de teste; de cima: vistas de treino não usadas, entre 50° e 89°. "
              r"As correções são aplicadas ao mesmo modelo treinado, sem novo treino. Em negrito, o melhor "
              r"entre as variantes desse modelo.}",
              rf"\label{{tab:principal{int(banda)}}}",
              r"\begin{tabular}{l ccc ccc ccc}", r"\toprule",
              rf" & \multicolumn{{3}}{{c}}{{Dentro da faixa ($<{banda:g}$°)}} & "
              rf"\multicolumn{{3}}{{c}}{{Acima ({banda:g}°--47°)}} & \multicolumn{{3}}{{c}}{{De cima (50°--89°)}} \\",
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
              r"Variante & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & PSNR$\uparrow$ & SSIM$\uparrow$ & "
              r"LPIPS$\downarrow$ & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ \\", r"\midrule"]
    melhor = {}
    for g in GRUPOS:
        for m, sinal in (("psnr", 1), ("ssim", 1), ("lpips", -1)):
            vals = {v: media_cenas(pc, g, v, m) for v in var}
            vals = {v: x for v, x in vals.items() if x is not None}
            if vals:
                melhor[(g, m)] = max(vals, key=lambda v: sinal * vals[v])
    for v in var:
        cel = []
        for g in GRUPOS:
            for m, casas in (("psnr", 2), ("ssim", 3), ("lpips", 3)):
                x = media_cenas(pc, g, v, m)
                s = fmt(x, casas)
                cel.append(rf"\textbf{{{s}}}" if melhor.get((g, m)) == v and x is not None else s)
        linhas.append(f"{NOME_VAR[v]} & " + " & ".join(cel) + r" \\")
    if pc_ref:
        linhas.append(r"\midrule")
        parcial = len(pc_ref) < len(pc)
        if parcial:  # média do 3DGS da faixa só nas cenas que têm referência, para comparar par a par
            pc_sub = {c: pc[c] for c in pc_ref if c in pc}
            cel = []
            for g in GRUPOS:
                for m, casas in (("psnr", 2), ("ssim", 3), ("lpips", 3)):
                    cel.append(fmt(media_cenas(pc_sub, g, "original", m), casas))
            linhas.append(rf"\textit{{3DGS, mesmas {len(pc_sub)} cenas}} & " + " & ".join(cel) + r" \\")
        cel = []
        for g in GRUPOS:
            for m, casas in (("psnr", 2), ("ssim", 3), ("lpips", 3)):
                cel.append(fmt(media_cenas(pc_ref, g, "original", m), casas) if g != "cima" else "n/a")
        rot = r"Referência: treino com o hemisfério todo" + (f" ({len(pc_ref)} cenas)" if parcial else "")
        linhas.append(rf"\textit{{{rot}}} & " + " & ".join(cel) + r" \\")
    linhas += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(linhas) + "\n"


def tabela_cenas(pc: dict, banda: float) -> str:
    linhas = [r"\begin{table}[t]", r"\centering",
              rf"\caption{{PSNR (dB) por cena, treino abaixo de {banda:g}°: 3DGS original e com a projeção por "
              r"visibilidade ($\Delta$ = ganho).}",
              rf"\label{{tab:cenas{int(banda)}}}", r"\setlength{\tabcolsep}{4pt}",
              r"\begin{tabular}{l ccc ccc}", r"\toprule",
              r" & \multicolumn{3}{c}{Acima} & \multicolumn{3}{c}{De cima} \\",
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
              r"Cena & orig. & proj. & $\Delta$ & orig. & proj. & $\Delta$ \\", r"\midrule"]
    for c in CENAS:
        if c not in pc:
            continue
        cel = []
        for g in ("acima", "cima"):
            o, p = pc[c][g]["original"]["psnr"], pc[c][g]["visib"]["psnr"]
            cel += [fmt(o), fmt(p), ("+" if p - o >= 0 else "") + fmt(p - o)]
        linhas.append(f"\\textit{{{c}}} & " + " & ".join(cel) + r" \\")
    linhas += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(linhas) + "\n"


def tabela_lambda(pc: dict) -> str:
    var = [("global", r"global, $\lambda=10^{-2}$"), ("visib_l0.001", r"visibilidade, $\lambda=10^{-3}$"),
           ("visib", r"visibilidade, $\lambda=10^{-2}$"), ("visib_l0.1", r"visibilidade, $\lambda=10^{-1}$")]
    linhas = [r"\begin{table}[t]", r"\centering",
              r"\caption{Sensibilidade à regularização $\lambda$ (PSNR médio em dB, faixa de 25°).}",
              r"\label{tab:lambda}", r"\begin{tabular}{l ccc}", r"\toprule",
              r"Projeção & Faixa & Acima & De cima \\", r"\midrule"]
    o = [fmt(media_cenas(pc, g, "original", "psnr")) for g in GRUPOS]
    linhas.append(r"sem projeção (original) & " + " & ".join(o) + r" \\")
    for v, nome in var:
        linhas.append(f"{nome} & " + " & ".join(fmt(media_cenas(pc, g, v, "psnr")) for g in GRUPOS) + r" \\")
    linhas += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(linhas) + "\n"


def curva(res: dict, banda: float, destino: Path, res_ref: dict | None = None) -> None:
    bins = np.arange(5, 95, 5)
    fig, ax = plt.subplots(figsize=(3.5, 2.3), dpi=200)
    estilos = {"original": ("#c0392b", "-"), "dc": ("#7f8c8d", ":"), "global": ("#e67e22", "--"),
               "visib": ("#2e86c1", "-")}
    for v, (cor, ls) in estilos.items():
        xs, ys = [], []
        for a, b in zip(bins[:-1], bins[1:]):
            por_cena = []
            for r in res.values():
                sel = [ln[v]["psnr"] for ln in r["por_vista"] if a <= ln["elev"] < b and v in ln]
                if sel:
                    por_cena.append(np.mean(sel))
            if len(por_cena) == len(res):
                xs.append((a + b) / 2)
                ys.append(np.mean(por_cena))
        ax.plot(xs, ys, ls, color=cor, lw=1.2, label=NOME_VAR[v])
    if res_ref and set(res_ref) >= set(res):  # só com as mesmas cenas, senão a média não é comparável
        xs, ys = [], []
        for a, b in zip(bins[:-1], bins[1:]):
            por_cena = [np.mean(s) for r in res_ref.values()
                        if (s := [ln["original"]["psnr"] for ln in r["por_vista"] if a <= ln["elev"] < b])]
            if len(por_cena) == len(res_ref):
                xs.append((a + b) / 2)
                ys.append(np.mean(por_cena))
        rot = "referência (hemisfério todo" + (f", {len(res_ref)} cenas)" if len(res_ref) < len(res) else ")")
        ax.plot(xs, ys, "-", color="#27ae60", lw=0.9, alpha=0.8, label=rot)
    ax.axvspan(0, banda, color="#2e86c1", alpha=0.08, lw=0)
    ax.text(banda / 2, ax.get_ylim()[0] if False else 0.02, "treino", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=6, color="#2e86c1")
    ax.set_xlim(5, 90)
    ax.set_xlabel("elevação da vista de avaliação (graus)", fontsize=7)
    ax.set_ylabel("PSNR médio (dB)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(fontsize=5.5, frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(destino / f"curva_elevacao_b{int(banda)}.pdf")
    fig.savefig(destino / f"curva_elevacao_b{int(banda)}.png")
    plt.close(fig)


def qualitativo(saidas: Path, rotulo: str, cenas: list[str], destino: Path, vista: str = "cima") -> None:
    cols = [("gt", "Referência"), ("original", "3DGS"), ("visib", "Projeção"), ("oraculo", "Oráculo de cor"),
            ("refgs", "Ref-Gaussian")]
    cenas = [c for c in cenas if (saidas / c / rotulo / "figs" / f"{vista}_gt.png").exists()]
    if not cenas:
        return
    # só as colunas que existem para todas as cenas escolhidas (Ref-Gaussian e oráculo podem faltar)
    cols = [(k, t) for k, t in cols if all((saidas / c / rotulo / "figs" / f"{vista}_{k}.png").exists() for c in cenas)]
    fig, axs = plt.subplots(len(cenas), len(cols), figsize=(1.4 * len(cols), 1.4 * len(cenas) + 0.2), dpi=300)
    axs = np.atleast_2d(axs)
    for i, c in enumerate(cenas):
        meta = json.loads((saidas / c / rotulo / "figs" / "vistas.json").read_text())[vista]
        for j, (k, titulo) in enumerate(cols):
            img = plt.imread(saidas / c / rotulo / "figs" / f"{vista}_{k}.png")
            h, w = img.shape[:2]
            img = img[int(h * 0.12):int(h * 0.88), int(w * 0.12):int(w * 0.88)]  # recorta a borda branca
            ax = axs[i, j]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.3)
            if i == 0:
                ax.set_title(titulo, fontsize=8, pad=3)
            if j == 0:
                ax.set_ylabel(f"{c}\n{meta['elev']:.0f}°", fontsize=7)
    fig.subplots_adjust(left=0.07, right=0.998, top=0.93, bottom=0.005, wspace=0.03, hspace=0.04)
    fig.savefig(destino / f"qualitativo_{vista}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(destino / f"qualitativo_{vista}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def decomposicao(saidas: Path, res: dict, banda: float) -> tuple[str, dict]:
    """Por cena: 3DGS, melhor correção sem treino, Ref-Gaussian e oráculo de cor, nos grupos acima e de cima.

    O oráculo cruzado é avaliado só nas vistas de índice ímpar do grupo; para comparar na mesma base, as
    outras colunas também são médias nessas vistas."""
    dados, linhas = {}, []
    for c in CENAS:
        if c not in res:
            continue
        f_or = saidas / c / f"b{banda}_sh3" / "oraculo.json"
        f_rg = saidas / c / f"refgs_b{banda}" / "por_vista.json"
        oraculo = json.loads(f_or.read_text()) if f_or.exists() else {}
        refgs = json.loads(f_rg.read_text()) if f_rg.exists() else []
        dados[c] = {}
        for g in ("acima", "cima"):
            sel = [ln for ln in res[c]["por_vista"] if grupo(ln, banda) == g]
            impares = sel[1::2]  # mesma ordem do oraculo.py (ordem dos transforms)
            nomes = {(ln["origem"], ln["nome"]) for ln in impares}
            d = {"original": float(np.mean([ln["original"]["psnr"] for ln in impares])),
                 "visib": float(np.mean([ln["visib"]["psnr"] for ln in impares]))}
            if g in oraculo:
                d["oraculo"] = oraculo[g]["oraculo_cruzado_impares"]
                d["original_check"] = oraculo[g]["original_impares"]
            rg = [ln["refgs"]["psnr"] for ln in refgs if (ln["origem"], ln["nome"]) in nomes]
            if rg:
                d["refgs"] = float(np.mean(rg))
            dados[c][g] = d
    cab = [r"\begin{table}[t]", r"\centering",
           rf"\caption{{PSNR (dB) fora da faixa de treino ($<{banda:g}$°), nas vistas de índice ímpar de cada grupo. "
           r"Oráculo: só a cor (SH) reajustada nas vistas de índice par do mesmo grupo, geometria congelada; é o teto "
           r"de uma correção só de cor. Ref-G.: Ref-Gaussian treinado com as mesmas vistas.}",
           rf"\label{{tab:decomp{int(banda)}}}", r"\setlength{\tabcolsep}{3pt}", r"\footnotesize",
           r"\begin{tabular}{l cccc cccc}", r"\toprule",
           r" & \multicolumn{4}{c}{Acima} & \multicolumn{4}{c}{De cima} \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
           r"Cena & 3DGS & proj. & Ref-G. & oráculo & 3DGS & proj. & Ref-G. & oráculo \\", r"\midrule"]
    medias = {g: {k: [] for k in ("original", "visib", "refgs", "oraculo")} for g in ("acima", "cima")}
    for c, d in dados.items():
        cel = []
        for g in ("acima", "cima"):
            for k in ("original", "visib", "refgs", "oraculo"):
                x = d[g].get(k)
                cel.append(fmt(x))
                if x is not None:
                    medias[g][k].append(x)
        linhas.append(f"\\textit{{{c}}} & " + " & ".join(cel) + r" \\")
    def linha_media(cenas: list[str], rotulo: str, so_completas: bool) -> str:
        cel = []
        for g in ("acima", "cima"):
            for k in ("original", "visib", "refgs", "oraculo"):
                xs = [dados[c][g][k] for c in cenas if k in dados[c][g]]
                cel.append(fmt(float(np.mean(xs))) if xs and (len(xs) == len(cenas) or not so_completas) else "--")
        return f"{rotulo} & " + " & ".join(cel) + r" \\"

    todas = list(dados)
    com_rg = [c for c in todas if "refgs" in dados[c]["cima"]]
    rod = [r"\midrule", linha_media(todas, "Média", True)]
    nota = []
    if com_rg and len(com_rg) < len(todas):  # Ref-Gaussian só em parte das cenas: média separada, mesma base
        rod.append(linha_media(com_rg, rf"Média$^*$", True))
        nota = [r"\vspace{2pt}", rf"{{\scriptsize $^*$ só nas cenas com Ref-Gaussian ({', '.join(com_rg)}).}}"]
    rod += [r"\bottomrule", r"\end{tabular}"] + nota + [r"\end{table}"]
    return "\n".join(cab + linhas + rod) + "\n", dados


def tabela_oraculo_oposto(saidas: Path, res: dict, banda: float) -> tuple[str, dict]:
    """Controle do oráculo: cor ajustada nas vistas de azimute < 0 do grupo, avaliada nas de azimute >= 0."""
    dados = {}
    for c in CENAS:
        f = saidas / c / f"b{banda:g}_sh3" / "oraculo.json"
        if c in res and f.exists():
            o = json.loads(f.read_text())
            if all("oraculo_lado_oposto" in o.get(g, {}) for g in ("acima", "cima")):
                dados[c] = {g: {"original": o[g]["original_lado_b"], "oraculo": o[g]["oraculo_lado_oposto"],
                                "n": o[g]["n_lado_b"], "todas": o[g]["oraculo_todas"],
                                "original_todas": o[g]["original"]} for g in ("acima", "cima")}
    if not dados:
        return "", {}
    linhas = [r"\begin{table}[t]", r"\centering",
              rf"\caption{{Controle do oráculo de cor (PSNR em dB, treino abaixo de {banda:g}°): a cor é reajustada "
              r"só nas vistas de um lado do objeto (azimute negativo) e avaliada nas do lado oposto, que nenhuma "
              r"vista do ajuste mostrou. A geometria continua congelada.}",
              rf"\label{{tab:oposto{int(banda)}}}", r"\setlength{\tabcolsep}{4pt}",
              r"\begin{tabular}{l ccc ccc}", r"\toprule",
              r" & \multicolumn{3}{c}{Acima} & \multicolumn{3}{c}{De cima} \\",
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
              r"Cena & 3DGS & oráculo & $\Delta$ & 3DGS & oráculo & $\Delta$ \\", r"\midrule"]
    for c, d in dados.items():
        cel = []
        for g in ("acima", "cima"):
            cel += [fmt(d[g]["original"]), fmt(d[g]["oraculo"]),
                    ("+" if d[g]["oraculo"] >= d[g]["original"] else "") + fmt(d[g]["oraculo"] - d[g]["original"])]
        linhas.append(f"\\textit{{{c}}} & " + " & ".join(cel) + r" \\")
    cel = []
    for g in ("acima", "cima"):
        o = float(np.mean([d[g]["original"] for d in dados.values()]))
        x = float(np.mean([d[g]["oraculo"] for d in dados.values()]))
        cel += [fmt(o), fmt(x), ("+" if x >= o else "") + fmt(x - o)]
    linhas += [r"\midrule", r"Média & " + " & ".join(cel) + r" \\", r"\bottomrule", r"\end{tabular}",
               r"\end{table}"]
    return "\n".join(linhas) + "\n", dados


def tabela_geometria(saidas: Path, banda: float) -> tuple[str, dict]:
    """IoU da silhueta e erro angular das normais por grupo (geometria.py), para o modelo da faixa e, se houver,
    para o modelo de referência treinado com o hemisfério todo."""
    dados, ref = {}, {}
    for c in CENAS:
        f = saidas / c / f"b{banda:g}_sh3" / "geometria.json"
        if f.exists():
            dados[c] = json.loads(f.read_text())["resumo"]
        f = saidas / c / "completo_sh3" / "geometria.json"
        if f.exists():
            ref[c] = json.loads(f.read_text())["resumo"]
    if not dados:
        return "", {}
    gs = ("faixa", "acima", "cima")
    linhas = [r"\begin{table}[t]", r"\centering",
              rf"\caption{{Geometria medida contra a referência do Shiny Blender, treino abaixo de {banda:g}°: IoU da "
              r"silhueta e erro angular médio (graus) das normais derivadas da profundidade renderizada. F: dentro da "
              r"faixa; A: acima; C: de cima.}",
              rf"\label{{tab:geometria{int(banda)}}}", r"\setlength{\tabcolsep}{4pt}", r"\footnotesize",
              r"\begin{tabular}{l ccc ccc}", r"\toprule",
              r" & \multicolumn{3}{c}{IoU da silhueta$\uparrow$} & \multicolumn{3}{c}{Erro da normal (°)$\downarrow$} \\",
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
              r"Cena & F & A & C & F & A & C \\", r"\midrule"]
    for c, d in dados.items():
        cel = [fmt(d[g]["iou"], 3) for g in gs] + [fmt(d[g]["normal_erro_medio"], 1) for g in gs]
        linhas.append(f"\\textit{{{c}}} & " + " & ".join(cel) + r" \\")
    med = {g: {k: float(np.mean([d[g][k] for d in dados.values()])) for k in ("iou", "normal_erro_medio")} for g in gs}
    cel = [fmt(med[g]["iou"], 3) for g in gs] + [fmt(med[g]["normal_erro_medio"], 1) for g in gs]
    linhas += [r"\midrule", r"Média & " + " & ".join(cel) + r" \\"]
    if ref:
        comuns = [c for c in ref if c in dados]
        mr = {g: {k: float(np.mean([ref[c][g][k] for c in comuns])) for k in ("iou", "normal_erro_medio")} for g in gs}
        if len(comuns) < len(dados):  # mesma média do modelo da faixa, só nas cenas com referência
            ms = {g: {k: float(np.mean([dados[c][g][k] for c in comuns])) for k in ("iou", "normal_erro_medio")}
                  for g in gs}
            cel = [fmt(ms[g]["iou"], 3) for g in gs] + [fmt(ms[g]["normal_erro_medio"], 1) for g in gs]
            linhas.append(rf"Mesmas {len(comuns)} cenas & " + " & ".join(cel) + r" \\")
        cel = [fmt(mr[g]["iou"], 3) for g in gs] + [fmt(mr[g]["normal_erro_medio"], 1) for g in gs]
        linhas.append(rf"Treino completo$^\dagger$ & " + " & ".join(cel) + r" \\")
    linhas += [r"\bottomrule", r"\end{tabular}"]
    if ref:
        linhas += [r"\vspace{2pt}",
                   rf"{{\scriptsize $^\dagger$ modelo treinado com as 100 vistas do hemisfério, média de {len(ref)} cenas "
                   r"(as vistas de cima fazem parte do treino dele).}"]
    linhas.append(r"\end{table}")
    return "\n".join(linhas) + "\n", {"b": dados, "completo": ref}


def tabela_normais(saidas: Path, banda: float) -> tuple[str, dict]:
    """Erro médio das normais (graus): 3DGS contra Ref-Gaussian, treinados na faixa ou com o hemisfério todo.
    Cada bloco compara só cenas que têm todos os modelos do bloco."""
    rot = {"3dgs_b": f"b{banda:g}_sh3", "3dgs_c": "completo_sh3", "refgs_b": f"refgs_b{banda:g}",
           "refgs_c": "refgs_completo"}
    g = {k: {} for k in rot}
    for c in CENAS:
        for k, r in rot.items():
            f = saidas / c / r / "geometria.json"
            if f.exists():
                g[k][c] = json.loads(f.read_text())["resumo"]
    nome = {"3dgs_b": rf"3DGS, faixa {banda:g}° (prof.)", "3dgs_b_eixo": rf"3DGS, faixa {banda:g}° (eixo)",
            "3dgs_c": r"3DGS, hemisfério todo$^\dagger$ (prof.)", "3dgs_c_eixo": r"3DGS, hemisfério todo$^\dagger$ (eixo)",
            "refgs_b": rf"Ref-Gaussian, faixa {banda:g}°", "refgs_c": r"Ref-Gaussian, hemisfério todo"}
    blocos = [["3dgs_b", "3dgs_b_eixo", "3dgs_c", "3dgs_c_eixo", "refgs_c"], ["3dgs_b", "3dgs_b_eixo", "refgs_b"]]
    # a normal pelo eixo mais curto sai do mesmo geometria.json do 3DGS
    for k in ("3dgs_b", "3dgs_c"):
        g[k + "_eixo"] = {c: {gr: {"n": d[gr]["n"], "normal_erro_medio": d[gr]["eixo_erro_medio"]} for gr in GRUPOS}
                          for c, d in g[k].items() if "eixo_erro_medio" in d["faixa"]}
    linhas = [r"\begin{table}[t]", r"\centering",
              r"\caption{Erro angular médio das normais (graus) contra a referência. No 3DGS, a normal é derivada da "
              r"profundidade renderizada (prof.) ou é o eixo mais curto de cada Gaussiana, composto por alfa (eixo); "
              r"no Ref-Gaussian, é a normal que o próprio modelo usa no sombreamento. "
              r"Cada bloco usa só as cenas em que todos os seus modelos existem.}",
              rf"\label{{tab:normais{int(banda)}}}", r"\setlength{\tabcolsep}{5pt}", r"\footnotesize",
              r"\begin{tabular}{l ccc}", r"\toprule",
              r"Modelo & Faixa & Acima & De cima \\"]
    out = {}
    for b in blocos:
        if not all(g[k] for k in b):
            continue
        comuns = [c for c in CENAS if all(c in g[k] for k in b)]
        linhas += [r"\midrule", rf"\multicolumn{{4}}{{l}}{{\textit{{{len(comuns)} cenas}}}} \\"]
        for k in b:
            cel = []
            for gr in GRUPOS:
                xs = [g[k][c][gr]["normal_erro_medio"] for c in comuns if g[k][c][gr]["n"] > 0]
                v = float(np.mean(xs)) if len(xs) == len(comuns) else None
                out.setdefault(f"{k}_{len(comuns)}", {})[gr] = v
                cel.append(fmt(v, 1) if v is not None else "n/a")
            linhas.append(f"{nome[k]} & " + " & ".join(cel) + r" \\")
    if len(linhas) == 9:
        return "", {}
    linhas += [r"\bottomrule", r"\end{tabular}", r"\vspace{2pt}",
               r"{\scriptsize $^\dagger$ as vistas de cima fazem parte do treino deste modelo.}", r"\end{table}"]
    return "\n".join(linhas) + "\n", out


NOME_CUSTO = {"original": "3DGS original (grau 3)", "grau0": "Truncamento, grau 0", "grau1": "Truncamento, grau 1",
              "grau2": "Truncamento, grau 2", "adaptativo": r"Grau adaptativo~\cite{papantonakis2024reducing}",
              "global": "Projeção global", "global_adapt": "Projeção global + grau adapt.",
              "visib": "Projeção por visibilidade", "visib_adapt": "Projeção por visib. + grau adapt."}


def tabela_custo(saidas: Path, banda: float) -> tuple[str, dict]:
    """Correções sem retreino: tempo, tamanho e qualidade por grupo (média das cenas com comparacao.json)."""
    dados = {}
    for c in CENAS:
        f = saidas / c / f"b{banda:g}_sh3" / "comparacao.json"
        if f.exists():
            dados[c] = json.loads(f.read_text())
    if not dados:
        return "", {}
    var = [v for v in NOME_CUSTO if all(v in d["mb"] for d in dados.values())]
    out = {}
    for v in var:
        m = {"tempo_s": float(np.mean([d["tempo_s"].get(v, 0.0) for d in dados.values()])),
             "mb": float(np.mean([d["mb"][v] for d in dados.values()]))}
        for g in GRUPOS:
            por_cena = [np.mean([ln[v][k] for ln in d["por_vista"] if grupo(ln, banda) == g])
                        for d in dados.values() for k in ("psnr",)]
            m[f"psnr_{g}"] = float(np.mean(por_cena))
        m["lpips_cima"] = float(np.mean([np.mean([ln[v]["lpips"] for ln in d["por_vista"] if grupo(ln, banda) == "cima"])
                                         for d in dados.values()]))
        out[v] = m
    melhor = {k: max(var, key=lambda v: out[v][k]) for k in ("psnr_faixa", "psnr_acima", "psnr_cima")}
    melhor["lpips_cima"] = min(var, key=lambda v: out[v]["lpips_cima"])
    linhas = [r"\begin{table*}[t]", r"\centering",
              rf"\caption{{Correções aplicadas ao modelo já treinado (treino abaixo de {banda:g}°), sem retreino: tempo "
              r"na GPU, tamanho médio do modelo em precisão simples e qualidade por grupo, média de "
              rf"{len(dados)} cenas. Em negrito, o melhor de cada coluna de qualidade. Para comparação, o retreino com "
              r"vistas de cima chega a 28,6 e 29,1~dB na faixa e acima, e o Ref-Gaussian exige um novo treino "
              r"cerca de 1,9 vez mais longo que o do 3DGS.}",
              rf"\label{{tab:custo{int(banda)}}}", r"\setlength{\tabcolsep}{4pt}",
              r"\begin{tabular}{l cc ccc c}", r"\toprule",
              r" & \multicolumn{2}{c}{Custo} & \multicolumn{3}{c}{PSNR (dB)$\uparrow$} & LPIPS$\downarrow$ \\",
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-7}",
              rf"Correção & Tempo (s) & Tamanho (MB) & Faixa & Acima & De cima & De cima \\", r"\midrule"]
    for v in var:
        m = out[v]
        cel = [fmt(m["tempo_s"], 2) if v != "original" else "--", fmt(m["mb"], 1)]
        for k, casas in (("psnr_faixa", 2), ("psnr_acima", 2), ("psnr_cima", 2), ("lpips_cima", 3)):
            s = fmt(m[k], casas)
            cel.append(rf"\textbf{{{s}}}" if melhor[k] == v else s)
        linhas.append(f"{NOME_CUSTO[v]} & " + " & ".join(cel) + r" \\")
        if v == "original":
            linhas.append(r"\midrule")
    linhas += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(linhas) + "\n", out


ROTULO_CURTO = {"grau0": "grau 0", "grau1": "grau 1", "grau2": "grau 2", "adaptativo": "grau adapt.",
                "global": "proj. global", "global_adapt": "global + adapt.", "visib": "proj. visib.",
                "visib_adapt": "visib. + adapt."}


def figura_custo(custo: dict, destino: Path, banda: float) -> None:
    """Variação de PSNR (contra o 3DGS original) por tamanho do modelo, na faixa e acima dela."""
    if not custo or "original" not in custo:
        return
    o = custo["original"]
    # compromisso: o que a correção custa na faixa filmada (x) contra o que ganha acima dela (y);
    # o rótulo traz o tamanho. Projeções com e sem grau adaptativo ficam quase no mesmo ponto: um rótulo só.
    desl = {"grau0": (5, 3), "grau1": (4, 2), "grau2": (4, 3), "adaptativo": (-4, -8),
            "global": (-4, -9), "visib": (-5, -9)}
    nomes = {"grau0": "grau 0", "grau1": "grau 1", "grau2": "grau 2", "adaptativo": "grau adaptativo",
             "global": "proj. global (± adapt.)", "visib": "proj. visibilidade (± adapt.)"}
    fig, ax = plt.subplots(figsize=(3.5, 2.3), dpi=200)
    ax.axhline(0, color="#7f8c8d", lw=0.6)
    ax.axvline(0, color="#7f8c8d", lw=0.6)
    for v, m in custo.items():
        if v == "original":
            continue
        proj = v.startswith(("global", "visib"))
        x, y = m["psnr_faixa"] - o["psnr_faixa"], m["psnr_acima"] - o["psnr_acima"]
        ax.scatter(x, y, s=14, color="#2e86c1" if proj else "#7f8c8d", zorder=3)
        if v in nomes:
            dx, dy = desl[v]
            ax.annotate(f"{nomes[v]}, {m['mb']:.0f} MB", (x, y), fontsize=5, xytext=(dx, dy),
                        textcoords="offset points", ha="right" if dx < 0 else "left", color="#2c3e50")
    ax.scatter(0, 0, s=16, marker="s", color="#c0392b", zorder=4)
    ax.annotate(f"3DGS original, {o['mb']:.0f} MB", (0, 0), fontsize=5, xytext=(-4, 5), textcoords="offset points",
                ha="right", color="#c0392b")
    ax.set_xlabel(r"$\Delta$ PSNR dentro da faixa (dB)", fontsize=6)
    ax.set_ylabel(r"$\Delta$ PSNR acima da faixa (dB)", fontsize=6)
    ax.tick_params(labelsize=5.5)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_xlim(min(m["psnr_faixa"] - o["psnr_faixa"] for m in custo.values()) - 0.15, 0.25)
    fig.tight_layout()
    fig.savefig(destino / f"custo_qualidade_b{int(banda)}.pdf")
    fig.savefig(destino / f"custo_qualidade_b{int(banda)}.png")
    plt.close(fig)


def tabela_refgs_iteracoes(saidas: Path, banda: float) -> tuple[str, dict]:
    """Ref-Gaussian treinado só na faixa (28 mil iterações) contra o completo em 30, 40 e 50 mil iterações, nas
    vistas de teste acima da faixa: separa o efeito das vistas de cima do efeito das iterações."""
    f = saidas / "refgs_iteracoes.json"
    if not f.exists():
        return "", {}
    it = json.loads(f.read_text())
    dados = {}
    for c in CENAS:
        if not all(f"{c}_{k}" in it for k in (30000, 40000, 50000)):
            continue
        d = {k: it[f"{c}_{k}"]["acima"] for k in (30000, 40000, 50000)}
        f_rg = saidas / c / f"refgs_b{banda:g}" / "por_vista.json"
        if f_rg.exists():
            rg = json.loads(f_rg.read_text())
            d["faixa"] = float(np.mean([ln["refgs"]["psnr"] for ln in rg if grupo(ln, banda) == "acima"]))
        dados[c] = d
    if not dados:
        return "", {}
    cols = ["faixa", 30000, 40000, 50000]
    linhas = [r"\begin{table}[t]", r"\centering",
              rf"\caption{{Ref-Gaussian nas vistas de teste acima de {banda:g}° (PSNR em dB): treinado só na faixa, com "
              r"28 mil iterações, contra o modelo treinado com todas as alturas avaliado nos \textit{checkpoints} de 30, "
              r"40 e 50 mil iterações. A diferença entre as colunas do modelo completo mede o efeito das iterações; "
              r"a diferença para a primeira coluna, o efeito das vistas de cima.}",
              rf"\label{{tab:refgsiter{int(banda)}}}", r"\setlength{\tabcolsep}{4pt}", r"\footnotesize",
              r"\begin{tabular}{l c ccc}", r"\toprule",
              r" & Faixa & \multicolumn{3}{c}{Todas as alturas} \\", r"\cmidrule(lr){2-2}\cmidrule(lr){3-5}",
              r"Cena & 28 mil & 30 mil & 40 mil & 50 mil \\", r"\midrule"]
    for c, d in dados.items():
        linhas.append(f"\\textit{{{c}}} & " + " & ".join(fmt(d.get(k)) for k in cols) + r" \\")
    com_faixa = [c for c in dados if "faixa" in dados[c]]
    med = {k: float(np.mean([dados[c][k] for c in com_faixa])) for k in cols} if com_faixa else {}
    if com_faixa:
        rot = "Média" if len(com_faixa) == len(dados) else rf"Média ({len(com_faixa)} cenas)"
        linhas += [r"\midrule", f"{rot} & " + " & ".join(fmt(med[k]) for k in cols) + r" \\"]
    linhas += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(linhas) + "\n", {"por_cena": dados, "media": med}


def tabela_refgs_oraculos(saidas: Path, banda: float) -> tuple[str, dict]:
    """Oráculos no Ref-Gaussian (refgs_oraculo.py): PSNR nas vistas de cima de índice ímpar, depois de reajustar
    com as de índice par só o material, só o ambiente, só a luz indireta ou as três. Ao lado, o 3DGS nas mesmas
    vistas, original e com os oráculos de cor e de geometria."""
    partes = ("material", "ambiente", "indireta", "aparencia")
    dados = {}
    for c in CENAS:
        f = saidas / c / f"refgs_b{banda:g}" / "oraculo_refgs.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        if not all(p in r for p in ("original",) + partes):
            continue
        d = {"refgs": r["original"]["impares"]["psnr"]}
        d.update({p: r[p]["impares"]["psnr"] for p in partes})
        f3 = saidas / c / f"b{banda:g}_sh3" / "oraculo.json"
        f32 = saidas / c / f"b{banda:g}_sh3" / "oraculo2.json"
        if f3.exists():
            o = json.loads(f3.read_text())["cima"]
            d["3dgs"], d["3dgs_cor"] = o["original_impares"], o["oraculo_cruzado_impares"]
        if f32.exists():
            d["3dgs_geo"] = json.loads(f32.read_text())["cima"]["oraculo_geometria_impares"]
        dados[c] = d
    if not dados:
        return "", {}
    cols = [("3dgs", "orig."), ("3dgs_cor", "cor"), ("3dgs_geo", "geom."), ("refgs", "orig."),
            ("material", "mat."), ("ambiente", "amb."), ("indireta", "ind."), ("aparencia", "todas")]
    linhas = [r"\begin{table}[t]", r"\centering",
              rf"\caption{{Oráculos de cima, treino abaixo de {banda:g}° (PSNR em dB nas vistas de cima de índice "
              r"ímpar; o ajuste usa as de índice par). No 3DGS, reajusta-se só a cor (SH) ou só a geometria. No "
              r"Ref-Gaussian, a geometria fica congelada e reajusta-se só o material (cor base, cor difusa, força do "
              r"reflexo, rugosidade, metalicidade), só o mapa de ambiente, só a luz indireta ou os três.}",
              rf"\label{{tab:refgsoraculos{int(banda)}}}", r"\setlength{\tabcolsep}{2.5pt}", r"\footnotesize",
              r"\begin{tabular}{l ccc ccccc}", r"\toprule",
              r" & \multicolumn{3}{c}{3DGS} & \multicolumn{5}{c}{Ref-Gaussian} \\",
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-9}",
              "Cena & " + " & ".join(n for _, n in cols) + r" \\", r"\midrule"]
    for c, d in dados.items():
        linhas.append(f"\\textit{{{c}}} & " + " & ".join(fmt(d.get(k)) for k, _ in cols) + r" \\")
    med = {k: float(np.mean([d[k] for d in dados.values() if k in d])) for k, _ in cols
           if all(k in d for d in dados.values())}
    linhas += [r"\midrule", "Média & " + " & ".join(fmt(med.get(k)) for k, _ in cols) + r" \\",
               r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(linhas) + "\n", {"por_cena": dados, "media": med}


def tabela_oraculos(saidas: Path, banda: float) -> tuple[str, dict]:
    """Vistas de cima: oráculos que reajustam só a cor (constante ou SH) ou só a geometria, com as vistas pares
    (avaliados nas ímpares) ou com um lado do objeto (avaliados no outro). Junta oraculo.json e oraculo2.json."""
    dados = {}
    for c in CENAS:
        p = saidas / c / f"b{banda:g}_sh3"
        if (p / "oraculo.json").exists() and (p / "oraculo2.json").exists():
            o1 = json.loads((p / "oraculo.json").read_text())["cima"]
            o2 = json.loads((p / "oraculo2.json").read_text())["cima"]
            dados[c] = {"imp_3dgs": o1["original_impares"], "imp_dc": o2["oraculo_cor_dc_impares"],
                        "imp_cor": o1["oraculo_cruzado_impares"], "imp_geo": o2["oraculo_geometria_impares"],
                        "op_3dgs": o1["original_lado_b"], "op_cor": o1["oraculo_lado_oposto"],
                        "op_geo": o2["oraculo_geometria_lado_oposto"]}
    if not dados:
        return "", {}
    cols = ("imp_3dgs", "imp_dc", "imp_cor", "imp_geo", "op_3dgs", "op_cor", "op_geo")
    linhas = [r"\begin{table}[t]", r"\centering",
              rf"\caption{{Oráculos nas vistas de cima (PSNR em dB, treino abaixo de {banda:g}°). Cada oráculo reajusta "
              r"só uma parte do modelo usando vistas de referência de cima: a cor constante de cada Gaussiana (DC), "
              r"todos os SH (cor) ou posições, escalas, rotações e opacidades (geom.). Ímpares: ajuste nas vistas de "
              r"índice par, avaliação nas ímpares. Lado oposto: ajuste num lado do objeto, avaliação no outro.}",
              rf"\label{{tab:oraculos{int(banda)}}}", r"\setlength{\tabcolsep}{3pt}", r"\footnotesize",
              r"\begin{tabular}{l cccc ccc}", r"\toprule",
              r" & \multicolumn{4}{c}{Ímpares} & \multicolumn{3}{c}{Lado oposto} \\",
              r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
              r"Cena & 3DGS & DC & cor & geom. & 3DGS & cor & geom. \\", r"\midrule"]
    for c, d in dados.items():
        linhas.append(f"\\textit{{{c}}} & " + " & ".join(fmt(d[k]) for k in cols) + r" \\")
    med = {k: float(np.mean([d[k] for d in dados.values()])) for k in cols}
    linhas += [r"\midrule", r"Média & " + " & ".join(fmt(med[k]) for k in cols) + r" \\",
               r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(linhas) + "\n", {"por_cena": dados, "media": med}


def estatistica(res: dict, banda: float) -> dict:
    """Diferença pareada por vista (visib - original) no grupo de cima: média, IC 95% bootstrap e % de vistas
    que melhoram."""
    out = {}
    rng = np.random.default_rng(0)
    for g in ("faixa", "acima", "cima"):
        d = np.array([ln["visib"]["psnr"] - ln["original"]["psnr"] for r in res.values() for ln in r["por_vista"]
                      if grupo(ln, banda) == g and "visib" in ln])
        if len(d) == 0:
            continue
        boot = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        out[g] = {"n": int(len(d)), "media": float(d.mean()), "ic95": [float(np.percentile(boot, 2.5)),
                                                                        float(np.percentile(boot, 97.5))],
                  "frac_melhora": float((d > 0).mean()), "max": float(d.max()), "min": float(d.min())}
    return out


def main():
    saidas, artigo = Path(sys.argv[1]), Path(sys.argv[2])
    (artigo / "figs").mkdir(parents=True, exist_ok=True)
    resumo = {}
    ref = carregar(saidas, "completo_sh3")
    for banda in (25, 15):
        res = carregar(saidas, f"b{banda}_sh3")
        if not res:
            continue
        var = ["original", "dc", "global", "visib", "visib_l0.001", "visib_l0.1"]
        pc = medias(res, banda, var)
        pc_ref = medias(ref, banda, ["original"]) if ref else {}
        (artigo / f"tab_principal_b{banda}.tex").write_text(tabela_principal(pc, pc_ref, banda, len(res)), "utf-8")
        (artigo / f"tab_cenas_b{banda}.tex").write_text(tabela_cenas(pc, banda), "utf-8")
        if banda == 25:
            (artigo / "tab_lambda.tex").write_text(tabela_lambda(pc), "utf-8")
        tex, dec = decomposicao(saidas, res, banda)
        (artigo / f"tab_decomp_b{banda}.tex").write_text(tex, "utf-8")
        tex_op, oposto = tabela_oraculo_oposto(saidas, res, banda)
        if tex_op:
            (artigo / f"tab_oposto_b{banda}.tex").write_text(tex_op, "utf-8")
        tex_or, oraculos = tabela_oraculos(saidas, banda)
        if tex_or:
            (artigo / f"tab_oraculos_b{banda}.tex").write_text(tex_or, "utf-8")
        tex_geo, geo = tabela_geometria(saidas, banda)
        if tex_geo:
            (artigo / f"tab_geometria_b{banda}.tex").write_text(tex_geo, "utf-8")
        tex_nor, normais_ = tabela_normais(saidas, banda)
        if tex_nor:
            (artigo / f"tab_normais_b{banda}.tex").write_text(tex_nor, "utf-8")
        tex_cu, custo = tabela_custo(saidas, banda)
        if tex_cu:
            (artigo / f"tab_custo_b{banda}.tex").write_text(tex_cu, "utf-8")
            figura_custo(custo, artigo / "figs", banda)
        tex_ro, refgs_or = tabela_refgs_oraculos(saidas, banda)
        if tex_ro:
            (artigo / f"tab_refgs_oraculos_b{banda}.tex").write_text(tex_ro, "utf-8")
        tex_it, refgs_it = tabela_refgs_iteracoes(saidas, banda)
        if tex_it and banda == 25:
            (artigo / f"tab_refgs_iteracoes_b{banda}.tex").write_text(tex_it, "utf-8")
        curva(res, banda, artigo / "figs", ref or None)
        qualitativo(saidas, f"b{banda}_sh3", ["car", "helmet", "toaster"], artigo / "figs")
        resumo[f"b{banda}"] = {
            "cenas": sorted(res), "estatistica": estatistica(res, banda), "decomposicao": dec,
            "oraculo_oposto": oposto, "geometria": geo, "oraculos": oraculos, "normais": normais_,
            "custo": custo if tex_cu else None, "refgs_oraculos": refgs_or if tex_ro else None,
            "medias": {g: {v: {m: media_cenas(pc, g, v, m) for m in ("psnr", "ssim", "lpips")} for v in var}
                       for g in GRUPOS},
            "ref": {g: {m: media_cenas(pc_ref, g, "original", m) for m in ("psnr", "ssim", "lpips")}
                    for g in ("faixa", "acima")} if pc_ref else None,
            "n_treino": {c: r["n_treino"] for c, r in res.items()},
            "n_gaussianas": {c: r["n_gaussianas"] for c, r in res.items()},
            "tempo_treino_min": {c: round(r["tempo_treino_s"] / 60, 1) for c, r in res.items()},
            "tempo_correcao_s": {c: r["tempo_correcao_s"] for c, r in res.items()},
            "frac_gaussianas_vistas": {c: r["frac_gaussianas_vistas"] for c, r in res.items()},
        }
    (artigo / "resumo_resultados.json").write_text(json.dumps(resumo, indent=1, ensure_ascii=False), "utf-8")
    print(json.dumps({k: {"estatistica": v["estatistica"], "medias_psnr": {
        g: {var: round(x["psnr"], 2) for var, x in gv.items() if x["psnr"] is not None}
        for g, gv in v["medias"].items()}} for k, v in resumo.items()}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
