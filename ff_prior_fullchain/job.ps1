param([ValidateSet('status','next','report')][string]$Action='status')
$TaskRepo = Split-Path -Parent $PSScriptRoot
$TaskPreviousEncoding = $env:PYTHONIOENCODING
$TaskExitCode = 0
Push-Location -LiteralPath $TaskRepo
try {
    $env:PYTHONIOENCODING = 'utf-8'
    & "$TaskRepo\.venv\Scripts\python.exe" -m ff_prior_fullchain.status_v2 $Action
    $TaskExitCode = $LASTEXITCODE
} finally {
    $env:PYTHONIOENCODING = $TaskPreviousEncoding
    Pop-Location
}
exit $TaskExitCode
