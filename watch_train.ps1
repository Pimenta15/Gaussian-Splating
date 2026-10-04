# Acompanha o treino do ref-gaussian ao vivo (Ctrl+C para sair).
# Uso: powershell -ExecutionPolicy Bypass -File watch_train.ps1 [-Log caminho\do\log.txt]
param(
    [string]$Log = "$PSScriptRoot\ref-gaussian\train_toaster_log.txt",
    [int]$Intervalo = 15
)

while ($true) {
    Clear-Host
    Write-Host "Log: $Log`n"
    $texto = Get-Content $Log -Tail 5 -Encoding UTF8 -ErrorAction SilentlyContinue | Out-String
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
        if ($it -lt 20000) { Write-Host "`nFase         : gaussianas (a partir de 20000 entra malha + raytracing, mais lenta)" }
        else { Write-Host "`nFase         : malha + raytracing (reflexos indiretos)" }
    }
    if ($texto -match 'Training complete') { Write-Host "`n*** TREINO CONCLUIDO ***" -ForegroundColor Green; break }
    if ($texto -match 'Traceback|Error') { Write-Host "`n*** ERRO - veja o final do log ***" -ForegroundColor Red; Write-Host $texto; break }
    $gpu = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>$null
    Write-Host "GPU (uso, memoria, temp): $gpu"
    Write-Host "`nAtualiza a cada $Intervalo s. Ctrl+C para sair (o treino continua)."
    Start-Sleep $Intervalo
}
