"""Atualiza tabelas, figuras e PDF do relatorio a partir dos treinos ja concluidos.

Uso (no ambiente ref-gaussian):  python atualizar_resultados.py
Le ref-gaussian/output/<cena>/metric.txt e os logs de treino; escreve
tab_replicacao.tex, tab_custo.tex, figs/curvas_psnr.png, paineis qualitativos,
recompila relatorio.pdf e regenera o zip do Overleaf.
"""
import os
import shutil
import sys
import re
import subprocess
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, "..", ".."))
RG = os.path.join(RAIZ, "ref-gaussian")
CENAS = ["ball", "car", "coffee", "helmet", "teapot", "toaster"]

# Tabela 1 de Yao et al. (ICLR 2025)
ARTIGO = {
    "psnr":  [37.01, 31.04, 34.63, 32.32, 47.16, 28.05],
    "ssim":  [0.981, 0.964, 0.976, 0.971, 0.998, 0.948],
    "lpips": [0.098, 0.033, 0.076, 0.049, 0.006, 0.074],
}
CASAS = {"psnr": 2, "ssim": 3, "lpips": 3}
PEND = r"\pend"
PDFLATEX = shutil.which("pdflatex") or os.path.expandvars(r"%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe")


def br(v, casas):
    return f"{v:.{casas}f}".replace(".", ",")


def log_treino(cena):
    for p in (os.path.join(RG, "logs", f"{cena}_train.txt"), os.path.join(RG, f"train_{cena}_log.txt")):
        if os.path.exists(p):
            return p
    return None


def metricas(cena):
    p = os.path.join(RG, "output", cena, "metric.txt")
    if not os.path.exists(p):
        return None
    m = re.search(r"psnr:([\d.]+), ssim:([\d.]+), lpips:([\d.]+), fps:([\d.]+)", open(p).read())
    if not m:
        return None
    psnr, ssim, lpips, fps = map(float, m.groups())
    tempo = None
    lg = log_treino(cena)
    if lg:
        t = re.findall(r"50001 \[(\d+):(\d+):(\d+)", open(lg, encoding="utf-8", errors="ignore").read())
        if t:
            h, mi, s = map(int, t[-1])
            tempo = h + mi / 60 + s / 3600
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips, "fps": fps, "tempo": tempo}


def curva(cena):
    """Pares (iteracao, PSNR-test) quando o valor de teste muda no log."""
    lg = log_treino(cena)
    if not lg:
        return []
    pts, ult = [], None
    for it, v in re.findall(r"(\d+)/50001 \[[^\]]*?PSNR-test=([\d.]+)", open(lg, encoding="utf-8", errors="ignore").read()):
        v = float(v)
        if v > 0 and v != ult:
            pts.append((int(it), v))
            ult = v
    return pts


def main():
    res = {c: metricas(c) for c in CENAS}
    feitas = [c for c in CENAS if res[c]]
    todas = len(feitas) == len(CENAS)

    # --- Tabela de replicacao ---
    linhas = []
    nomes = {"psnr": r"PSNR $\uparrow$", "ssim": r"SSIM $\uparrow$", "lpips": r"LPIPS $\downarrow$"}
    for k in ("psnr", "ssim", "lpips"):
        c_ = CASAS[k]
        art = [br(v, c_) for v in ARTIGO[k]] + [br(sum(ARTIGO[k]) / 6, c_)]
        rep = [br(res[c][k], c_) if res[c] else PEND for c in CENAS]
        rep.append(br(sum(res[c][k] for c in CENAS) / 6, c_) if todas else PEND)
        dif = []
        for i, c in enumerate(CENAS):
            if res[c] and k == "psnr":
                dif.append(f"{res[c][k] - ARTIGO[k][i]:+.2f}".replace(".", ","))
            else:
                dif.append("")
        linhas.append(rf"\multirow{{2}}{{*}}{{{nomes[k]}}} & Artigo & " + " & ".join(art) + r" \\")
        linhas.append(r" & Replicação & " + " & ".join(rep) + r" \\")
        if k == "psnr" and feitas:
            d = dif + ([f"{sum(res[c]['psnr'] for c in CENAS) / 6 - sum(ARTIGO['psnr']) / 6:+.2f}".replace(".", ",")] if todas else [""])
            linhas.append(r" & $\Delta$ & " + " & ".join(r"\footnotesize " + x if x else "" for x in d) + r" \\")
        linhas.append(r"\midrule" if k != "lpips" else "")
    open(os.path.join(AQUI, "tab_replicacao.tex"), "w", encoding="utf-8").write("\n".join(linhas) + "\n")

    # --- Tabela de custo ---
    lc = []
    for c in CENAS:
        r = res[c]
        if r:
            tempo = f"{int(r['tempo'])}h{round((r['tempo'] % 1) * 60):02d}min" if r["tempo"] else PEND
            lc.append(rf"\textit{{{c}}} & {tempo} & {br(r['fps'], 1)} \\")
        else:
            lc.append(rf"\textit{{{c}}} & {PEND} & {PEND} \\")
    open(os.path.join(AQUI, "tab_custo.tex"), "w", encoding="utf-8").write("\n".join(lc) + "\n")

    # --- Curvas de PSNR de teste ---
    plt.figure(figsize=(3.5, 2.0), dpi=200)
    for c in feitas:
        pts = curva(c)
        if pts:
            xs, ys = zip(*pts)
            plt.plot([x / 1000 for x in xs], ys, marker="o", ms=2.5, lw=1.2, label=c)
    for x, rot in ((18, "fim sombr. por Gaussiana"), (20, "início inter-reflexão")):
        plt.axvline(x, color="gray", ls="--", lw=0.7)
    plt.text(18.4, plt.ylim()[0] + 0.5, "18k / 20k", fontsize=6, color="gray")
    plt.xlabel("iteração (mil)", fontsize=7)
    plt.ylabel("PSNR de teste (dB)", fontsize=7)
    plt.tick_params(labelsize=6)
    plt.grid(alpha=0.3, lw=0.5)
    plt.legend(fontsize=6, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(AQUI, "figs", "curvas_psnr.png"))
    plt.close()

    # --- Paineis qualitativos ---
    for c in feitas:
        destino = os.path.join(AQUI, "figs", f"{c}_qualitativo.png")
        if not os.path.exists(destino):
            subprocess.run([sys.executable, os.path.join(AQUI, "figs", "make_figs.py"), c, "0"], check=False)

    # --- Compila e empacota ---
    for _ in range(2):
        subprocess.run([PDFLATEX, "-interaction=nonstopmode", "-halt-on-error", "relatorio.tex"],
                       cwd=AQUI, stdout=subprocess.DEVNULL, check=True)
    arquivos = ["relatorio.tex", "IEEEtran.cls", "tab_replicacao.tex", "tab_custo.tex"]
    arquivos += [os.path.join("figs", f) for f in os.listdir(os.path.join(AQUI, "figs")) if f.endswith(".png")]
    with zipfile.ZipFile(os.path.join(AQUI, "relatorio-ieee-overleaf.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for f in arquivos:
            z.write(os.path.join(AQUI, f), f)

    print("cenas concluidas:", ", ".join(feitas) or "nenhuma")
    for c in feitas:
        r = res[c]
        print(f"  {c:8} PSNR {r['psnr']:.2f}  SSIM {r['ssim']:.3f}  LPIPS {r['lpips']:.3f}  FPS {r['fps']:.1f}  tempo {r['tempo']}")


if __name__ == "__main__":
    main()
