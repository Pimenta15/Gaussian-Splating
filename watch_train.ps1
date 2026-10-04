# Acompanha o treino do ref-gaussian ao vivo (Ctrl+C para sair; o treino continua).
# Uso:
#   powershell -ExecutionPolicy Bypass -File watch_train.ps1            # segue a fila (logs\*_train.txt mais recente)
#   powershell -ExecutionPolicy Bypass -File watch_train.ps1 -Log <arquivo>   # segue um log especifico
param(
    [string]$Log = "",
    [int]$Intervalo = 15
)

$logs = "$PSScriptRoot\ref-gaussian\logs"
$fila = "$logs\queue.txt"
$cenas = "coffee", "helmet", "ball", "teapot", "car"

function Status-Fila {
    if (-not (Test-Path $fila)) { return }
    $q = Get-Content $fila -ErrorAction SilentlyContinue
    Write-Host "FILA:"
    foreach ($c in $cenas) {
        $l = $q | Where-Object { $_ -match " $c " } | Select-Object -Last 1
        if (-not $l) { $s = "aguardando" }
        elseif ($l -match "concluida") { $s = "concluida" }
        elseif ($l -match "FALHOU") { $s = "FALHOU" }
        elseif ($l -match "avaliacao") { $s = "avaliando..." }
        else { $s = "treinando..." }
        $m = "$PSScriptRoot\ref-gaussian\output\$c\metric.txt"
        $extra = ""
        if (Test-Path $m) {
            $t = Get-Content $m -Raw
            if ($t -match 'psnr:([\d\.]+), ssim:([\d\.]+), lpips:([\d\.]+)') {
                $extra = "  PSNR {0:N2}  SSIM {1:N3}  LPIPS {2:N3}" -f [double]$Matches[1], [double]$Matches[2], [double]$Matches[3]
            }
        }
        $cor = @{ "concluida" = "Green"; "FALHOU" = "Red"; "aguardando" = "DarkGray" }[$s]
        if (-not $cor) { $cor = "Yellow" }
        Write-Host ("  {0,-8} {1,-13}{2}" -f $c, $s, $extra) -ForegroundColor $cor
    }
    if ($q -match "FILA CONCLUIDA") { Write-Host "`n*** FILA CONCLUIDA ***" -ForegroundColor Green }
    Write-Host ""
}

while ($true) {
    Clear-Host
    Status-Fila
    $atual = $Log
    if (-not $atual) {
        $atual = Get-ChildItem "$logs\*_train.txt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1 -ExpandProperty FullName
    }
    if ($atual) {
        Write-Host "Log: $atual`n"
        $texto = Get-Content $atual -Tail 5 -Encoding UTF8 -ErrorAction SilentlyContinue | Out-String
        $m = [regex]::Matches($texto, '(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d\.]+)(it/s|s/it).*?Points=(\d+), PSNR-train=([\d\.]+), PSNR-test=([\d\.]+)')
        if ($m.Count) {
            $g = $m[$m.Count - 1].Groups
            $it = [int]$g[1].Value; $tot = [int]$g[2].Value - 1
            $pct = [math]::Round(100 * $it / $tot, 1)
            $barra = ('#' * [int]($pct / 2.5)).PadRight(40, '.')
            Write-Host ("[{0}] {1}%" -f $barra, $pct)
            Write-Host ("Iteracao     : {0} / {1}" -f $it, $tot)
            Write-Host ("Decorrido    : {0}" -f $g[3].Value)
            Write-Host ("Restante     : {0}  (estimativa pela velocidade atual)" -f $g[4].Value)
            Write-Host ("Velocidade   : {0} {1}" -f $g[5].Value, $g[6].Value)
            Write-Host ("Pontos       : {0}" -f $g[7].Value)
            Write-Host ("PSNR treino  : {0}   PSNR teste: {1}" -f $g[8].Value, $g[9].Value)
            if ($it -lt 20000) { Write-Host "Fase         : gaussianas (a partir de 20000 entra malha + raytracing)" }
            else { Write-Host "Fase         : malha + raytracing (reflexos indiretos)" }
        }
        if ($texto -match 'Traceback') { Write-Host "`n*** ERRO no log acima ***" -ForegroundColor Red }
        if ($Log -and $texto -match 'Training complete') { Write-Host "`n*** TREINO CONCLUIDO ***" -ForegroundColor Green; break }
    }
    $gpu = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>$null
    Write-Host "`nGPU (uso, memoria, temp): $gpu"
    Write-Host "Atualiza a cada $Intervalo s. Ctrl+C para sair (o treino continua)."
    if ((Test-Path $fila) -and ((Get-Content $fila) -match "FILA CONCLUIDA") -and -not $Log) { break }
    Start-Sleep $Intervalo
}
