param(
    [string]$Python='E:\v7work_20261010_01\.venv-v7\Scripts\python.exe'
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$outputRoot=Join-Path $projectRoot 'tracking_output\paper500_v2'
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
if (-not (Test-Path -LiteralPath $Python)) { throw "Python environment not found: $Python" }
function Start-RegisteredStage([string]$Stage,[string]$Module,[string[]]$Arguments,[string]$Completion) {
    $holdPath=Join-Path $outputRoot 'test_hold.json'
    if ($Stage -eq 'final' -and (Test-Path -LiteralPath $holdPath)) {
        $hold=Get-Content -LiteralPath $holdPath -Raw | ConvertFrom-Json
        if ($hold.active) {
            Write-Host "Final test awaits correction validation and configuration freeze: $($hold.reason)"
            return
        }
    }
    if ($Completion -and (Test-Path -LiteralPath (Join-Path $outputRoot $Completion))) {
        $status=Get-Content -LiteralPath (Join-Path $outputRoot $Completion) -Raw | ConvertFrom-Json
        if ($status.status -eq 'complete' -or $status.production_complete -eq $true) {
            Write-Host "$Stage already complete"
            return
        }
    }
    $pidPath=Join-Path $outputRoot "$Stage.pid"
    if (Test-Path -LiteralPath $pidPath) {
        $stagePid=[int](Get-Content -LiteralPath $pidPath -Raw).Trim()
        $existing=Get-CimInstance Win32_Process -Filter "ProcessId=$stagePid"
        if ($existing -and $existing.CommandLine -like "* $Module*") {
            Write-Host "$Stage already running (PID $stagePid)"
            return
        }
    }
    $stageProcess=Start-Process -FilePath $Python -ArgumentList (@('-m',$Module)+$Arguments) `
        -WorkingDirectory $projectRoot -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $outputRoot "$Stage.stdout.log") `
        -RedirectStandardError (Join-Path $outputRoot "$Stage.stderr.log") -PassThru
    $stageProcess.Id | Set-Content -LiteralPath $pidPath
    Write-Host "$Stage started (PID $($stageProcess.Id))"
}
Start-RegisteredStage 'produce' 'tracking_diffusion.round2_data' @('produce','--workers','4') 'teachers\summary.json'
Start-RegisteredStage 'train_all' 'tracking_diffusion.round2_train_all' @() 'training_stage.json'
Start-RegisteredStage 'screen' 'tracking_diffusion.round2_screen' @() 'screen_stage.json'
Start-RegisteredStage 'refine' 'tracking_diffusion.round2_refine' @() 'refine_stage.json'
Start-RegisteredStage 'validate' 'tracking_diffusion.round2_validate' @() 'validation_stage.json'
Start-RegisteredStage 'final' 'tracking_diffusion.round2_final' @() 'final_stage.json'
Write-Host 'Pipeline active. Live UI: http://127.0.0.1:8765/'
Write-Host 'Stage JSON and logs: tracking_output\paper500_v2'
Write-Host 'Do not edit experiment Python sources while an evaluation is running; snapshots enforce reproducibility.'
