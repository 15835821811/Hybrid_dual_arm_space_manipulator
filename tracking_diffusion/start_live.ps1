param(
    [string]$Python = '',
    [int]$Port = 8765,
    [switch]$NoBrowser,
    [ValidateSet('cpu','cuda')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$logFolder = Join-Path $projectRoot 'tracking_output\live'
New-Item -ItemType Directory -Path $logFolder -Force | Out-Null
$url = "http://127.0.0.1:$Port"
try {
    $health = Invoke-RestMethod "$url/api/state" -TimeoutSec 2
} catch {
    $health = $null
}
if ($null -ne $health) {
    if ($null -eq $health.checkpoint_available) { throw 'Port is used by a different service.' }
    if ($health.workflow -ne 'paper') { throw 'An older server is using this port. Restart that tracking server to load the paper workflow.' }
    Write-Host "Already running: $url"
    if (-not $NoBrowser) { Start-Process $url }
    exit 0
}
if (-not $Python) {
    $candidates = @(
        'E:\v7work_20261010_01\.venv-v7\Scripts\python.exe',
        (Join-Path $projectRoot '.venv-learning-cuda\Scripts\python.exe'),
        (Join-Path $projectRoot '.venv\Scripts\python.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            & $candidate -c 'import mujoco, numpy, scipy, torch' 2>$null
            if ($LASTEXITCODE -eq 0) { $Python = $candidate; break }
        }
    }
}
if (-not $Python -or -not (Test-Path -LiteralPath $Python)) {
    throw 'No working Python found. Supply -Python with an environment containing mujoco, numpy, scipy and torch.'
}
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$process = Start-Process -FilePath $Python -ArgumentList @('-m','tracking_diffusion.live_server','--port',"$Port",'--device',$Device) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logFolder 'server.stdout.log') -RedirectStandardError (Join-Path $logFolder 'server.stderr.log')
$process.Id | Set-Content (Join-Path $logFolder 'server.pid')
$ready = $false
for ($attempt=0; $attempt -lt 45; $attempt++) {
    if ($process.HasExited) { throw "Server stopped. Read $logFolder\server.stderr.log" }
    try { $null = Invoke-RestMethod "$url/api/state" -TimeoutSec 1; $ready = $true; break } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $ready) { throw "Server is still starting; see $logFolder\server.stderr.log" }
Write-Host "Ready: $url  (PID $($process.Id))"
Write-Host 'Ready for guided diffusion: 60 DDIM steps, final 18 guided steps. The page shows the loaded dataset and model.'
if (-not $NoBrowser) { Start-Process $url }
