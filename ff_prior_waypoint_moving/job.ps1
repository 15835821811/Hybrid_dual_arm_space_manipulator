param([ValidateSet("start","status","report","resume")][string]$Action="status")
$ErrorActionPreference="Stop"
$root=Split-Path $PSScriptRoot -Parent
$run=Join-Path $root "artifacts/v7_1/waypoint_moving_gate_001"
if ($Action -eq "status") {
    if (Test-Path "$run/status.json") { Get-Content "$run/status.json" } else { "NOT_STARTED" }
    exit
}
if ($Action -eq "report") {
    if (Test-Path "$PSScriptRoot/docs/ANALYSIS.md") { Get-Content "$PSScriptRoot/docs/ANALYSIS.md" }
    elseif (Test-Path "$run/summary.json") { Get-Content "$run/summary.json" }
    else { "Report pending; use status." }
    exit
}
if (Test-Path "$run/status.json") {
    $state=Get-Content "$run/status.json" -Raw | ConvertFrom-Json
    if ($state.status -in @("BLOCKED","FINISHED","FAILED","READY_FOR_TRAINING")) { "Terminal run preserved; no repeat work. Status: $($state.status)"; exit }
    if (Get-Process -Id $state.pid -ErrorAction SilentlyContinue) { "Worker alive: $($state.pid)"; exit }
    throw "Unknown interrupted run preserved; inspect logs before any new version."
}
New-Item -ItemType Directory -Force $run | Out-Null
if (Test-Path "$run/launch.json") { throw "Prior launch exists; inspect PID instead of duplicate start." }
$process=Start-Process -FilePath "$root/.venv/Scripts/python.exe" -ArgumentList "-m","ff_prior_waypoint_moving.worker" -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput "$run/worker.stdout.log" -RedirectStandardError "$run/worker.stderr.log" -PassThru
@{pid=$process.Id;time=(Get-Date -Format o)} | ConvertTo-Json | Set-Content "$run/launch.json"
"Started PID $($process.Id)"


