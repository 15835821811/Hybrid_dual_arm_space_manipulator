param([ValidateSet("start","status","report","resume")][string]$Action="status")
$ErrorActionPreference="Stop"
$root=Split-Path $PSScriptRoot -Parent
$run=Join-Path $root "artifacts/v7_1/waypoint_scene_001"
if ($Action -eq "status") {
    if (Test-Path "$run/status.json") { Get-Content "$run/status.json" } else { "NOT_STARTED" }
    if (Test-Path "$run/delivery.json") { Get-Content "$run/delivery.json" }
    exit
}
if ($Action -eq "report") {
    if (Test-Path "$PSScriptRoot/docs/ANALYSIS.md") { Get-Content "$PSScriptRoot/docs/ANALYSIS.md" }
    elseif (Test-Path "$run/summary.json") { Get-Content "$run/summary.json" }
    else { "Report pending. Use status." }
    exit
}
if (Test-Path "$run/status.json") {
    $state=Get-Content "$run/status.json" -Raw | ConvertFrom-Json
    if ($state.status -eq "FINISHED") { "Already finished; no repeat work."; exit }
    if (Get-Process -Id $state.pid -ErrorAction SilentlyContinue) { "Worker already alive: $($state.pid)"; exit }
    throw "Existing unfinished run preserved. No automatic replay of unknown work."
}
New-Item -ItemType Directory -Path $run -Force | Out-Null
$process=Start-Process -FilePath "$root/.venv/Scripts/python.exe" -ArgumentList "-m","ff_prior_waypoint_scene.experiment" -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput "$run/worker.stdout.log" -RedirectStandardError "$run/worker.stderr.log" -PassThru
"Started PID $($process.Id)"
