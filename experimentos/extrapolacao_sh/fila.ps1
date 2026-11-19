# Fila do experimento de extrapolação em elevação: uma execução por vez na GPU, retoma de onde parou.
#   powershell -ExecutionPolicy Bypass -File experimentos\extrapolacao_sh\fila.ps1
$exp   = "C:\Users\Pimenta\Desktop\Gaussian-Splating\experimentos\extrapolacao_sh"
$dados = "C:\Users\Pimenta\Desktop\Gaussian-Splating\ref-gaussian\data\ref_nerf"
$fonte = "C:\Users\Pimenta\Desktop\vitrine3d\services\worker\vitrine_worker"
$vit   = "C:\Users\Pimenta\Desktop\vitrine3d\outputs"
New-Item -ItemType Directory -Force "$exp\saidas" | Out-Null
$log = "$exp\saidas\fila.log"
$base = "docker run --rm --gpus all -v `"$dados`:/data:ro`" -v `"$dados`:/dados/ref_nerf:ro`" -v `"$exp\dados_banda`:/dados/banda`" -v `"$exp`:/exp`" -v `"$fonte`:/app/vitrine_worker:ro`" -v vitrine3d-cache:/cache -e TORCH_HOME=/cache/torch -w /exp vitrine3d-worker:dev"

function Esperar-GPU {  # outra execução (de uma fila anterior) ainda na GPU
    while (@(docker ps -q --filter ancestor=vitrine3d-worker:dev).Count -gt 0) { Start-Sleep 20 }
}

$cenas = "car", "helmet", "toaster", "coffee", "teapot", "ball"
$jobs = @()
foreach ($c in $cenas) { $jobs += ,@("3dgs", $c, 25) }
foreach ($c in $cenas) { $jobs += ,@("oraculo", $c, 25) }
$jobs += ,@("porsche", "", 0)
foreach ($c in $cenas) { $jobs += ,@("geometria", $c, 25) }
foreach ($c in $cenas) { $jobs += ,@("oraculo2", $c, 25) }
foreach ($c in "car", "helmet", "toaster") { $jobs += ,@("3dgs", $c, 90); $jobs += ,@("geometria", $c, 90) }
$jobs += ,@("refgs", "car", 25)
foreach ($c in $cenas) { $jobs += ,@("eixo", $c, 25) }
foreach ($c in "car", "helmet", "toaster") { $jobs += ,@("eixo", $c, 90) }
$jobs += ,@("refgs", "helmet", 25)
$jobs += ,@("refgs", "toaster", 25)
foreach ($c in "coffee", "teapot", "ball") { $jobs += ,@("3dgs", $c, 90); $jobs += ,@("geometria", $c, 90) }
foreach ($c in $cenas) { $jobs += ,@("3dgs", $c, 15) }
foreach ($c in $cenas) { $jobs += ,@("oraculo", $c, 15) }

foreach ($j in $jobs) {
    $tipo, $cena, $banda = $j
    if ($tipo -eq "porsche") {
        $ws = "$vit\workspaces\exp_porsche_artigo"
        if (Test-Path "$exp\saidas\porsche\resultado.json") { continue }
        if (-not (Test-Path "$ws\colmap")) {
            New-Item -ItemType Directory -Force $ws | Out-Null
            Copy-Item -Recurse "$vit\workspaces\exp_porsche_longo\colmap" "$ws\colmap"
            Copy-Item -Recurse "$vit\workspaces\exp_porsche_longo\quadros" "$ws\quadros"
        }
        Esperar-GPU
        if (-not (Test-Path "$ws\modelo\cena_sem_projecao.ply")) {  # o treino pode ter sido de uma fila anterior
            "$(Get-Date -Format 'dd/MM HH:mm') porsche (real, 30k, sem projecao salvo)" | Tee-Object -Append $log
            cmd /c "docker run --rm --gpus all -v `"$vit`:/outputs`" -v `"$fonte`:/app/vitrine_worker:ro`" -v vitrine3d-cache:/cache -e TORCH_HOME=/cache/torch -e MLFLOW_TRACKING_URI=file:/outputs/mlruns vitrine3d-worker:dev python -m vitrine_worker.reconstruir --video x --trabalho /outputs/workspaces/exp_porsche_artigo --reusar-colmap --config {\`"salvar_sem_projecao\`":true} > `"$ws\run.log`" 2>&1"
        }
        New-Item -ItemType Directory -Force "$exp\saidas\porsche" | Out-Null
        cmd /c "docker run --rm --gpus all -v `"$vit`:/outputs`" -v `"$exp`:/exp`" -v `"$fonte`:/app/vitrine_worker:ro`" -v vitrine3d-cache:/cache -w /exp vitrine3d-worker:dev python /exp/carro_real.py /outputs/workspaces/exp_porsche_artigo/modelo /exp/saidas/porsche > `"$exp\saidas\porsche\run.log`" 2>&1"
        Get-Content "$exp\saidas\porsche\run.log" -Tail 6 | Tee-Object -Append $log
        continue
    }
    if ($tipo -eq "oraculo") {
        $pasta = "$exp\saidas\$cena\b$($banda)_sh3"
        if ((Test-Path "$pasta\oraculo.json") -or -not (Test-Path "$pasta\params.pt")) { continue }
        Esperar-GPU
        if (Test-Path "$pasta\oraculo.json") { continue }
        "$(Get-Date -Format 'dd/MM HH:mm') oraculo $cena b$banda" | Tee-Object -Append $log
        cmd /c "$base python /exp/oraculo.py --cena $cena --banda $banda > `"$pasta\oraculo.log`" 2>&1"
        Get-Content "$pasta\oraculo.log" -Tail 2 | Tee-Object -Append $log
        continue
    }
    if ($tipo -eq "geometria") {
        $rot = if ($banda -lt 90) { "b$($banda)_sh3" } else { "completo_sh3" }
        $pasta = "$exp\saidas\$cena\$rot"
        if ((Test-Path "$pasta\geometria.json") -or -not (Test-Path "$pasta\params.pt")) { continue }
        Esperar-GPU
        if (Test-Path "$pasta\geometria.json") { continue }
        "$(Get-Date -Format 'dd/MM HH:mm') geometria $cena $rot" | Tee-Object -Append $log
        cmd /c "$base python /exp/geometria.py --cena $cena --banda 25 --rotulo $rot > `"$pasta\geometria.log`" 2>&1"
        Get-Content "$pasta\geometria.log" -Tail 1 | Tee-Object -Append $log
        continue
    }
    if ($tipo -eq "eixo") {  # refaz a geometria para somar a normal do eixo mais curto
        $rot = if ($banda -lt 90) { "b$($banda)_sh3" } else { "completo_sh3" }
        $pasta = "$exp\saidas\$cena\$rot"
        $tem = (Test-Path "$pasta\geometria.json") -and (Select-String -Quiet -Path "$pasta\geometria.json" -Pattern "eixo_erro_medio")
        if ($tem -or -not (Test-Path "$pasta\params.pt")) { continue }
        Esperar-GPU
        "$(Get-Date -Format 'dd/MM HH:mm') eixo $cena $rot" | Tee-Object -Append $log
        cmd /c "$base python /exp/geometria.py --cena $cena --banda 25 --rotulo $rot > `"$pasta\geometria.log`" 2>&1"
        Get-Content "$pasta\geometria.log" -Tail 1 | Tee-Object -Append $log
        continue
    }
    if ($tipo -eq "oraculo2") {
        $pasta = "$exp\saidas\$cena\b$($banda)_sh3"
        if ((Test-Path "$pasta\oraculo2.json") -or -not (Test-Path "$pasta\params.pt")) { continue }
        Esperar-GPU
        if (Test-Path "$pasta\oraculo2.json") { continue }
        "$(Get-Date -Format 'dd/MM HH:mm') oraculo2 $cena b$banda" | Tee-Object -Append $log
        cmd /c "$base python /exp/oraculo2.py --cena $cena --banda $banda > `"$pasta\oraculo2.log`" 2>&1"
        Get-Content "$pasta\oraculo2.log" -Tail 1 | Tee-Object -Append $log
        continue
    }
    if ($tipo -eq "refgs") {
        $pasta = "$exp\saidas\$cena\refgs_b$banda"
        if (Test-Path "$pasta\por_vista.json") { continue }
        Esperar-GPU
        if (Test-Path "$pasta\por_vista.json") { continue }
        New-Item -ItemType Directory -Force $pasta | Out-Null
        "$(Get-Date -Format 'dd/MM HH:mm') refgs $cena b$banda" | Tee-Object -Append $log
        cmd /c "$base python /exp/refgs_rodar.py /dados/banda/$($cena)_b$banda /exp/saidas/$cena/refgs_b$banda $banda 28000 > `"$pasta\run.log`" 2>&1"
        Get-Content "$pasta\run.log" -Tail 5 | Tee-Object -Append $log
        continue
    }
    $rot = if ($banda -lt 90) { "b$($banda)_sh3" } else { "completo_sh3" }
    if (Test-Path "$exp\saidas\$cena\$rot\resultado.json") { continue }
    Esperar-GPU
    if (Test-Path "$exp\saidas\$cena\$rot\resultado.json") { continue }  # a execução que estava na GPU era esta
    New-Item -ItemType Directory -Force "$exp\saidas\$cena\$rot" | Out-Null
    "$(Get-Date -Format 'dd/MM HH:mm') $cena $rot" | Tee-Object -Append $log
    cmd /c "$base python /exp/rodar.py --cena $cena --banda $banda --sh-grau 3 --saida /exp/saidas > `"$exp\saidas\$cena\$rot\run.log`" 2>&1"
    Get-Content "$exp\saidas\$cena\$rot\run.log" -Tail 4 | Tee-Object -Append $log
}
"$(Get-Date -Format 'dd/MM HH:mm') fila concluida" | Tee-Object -Append $log
