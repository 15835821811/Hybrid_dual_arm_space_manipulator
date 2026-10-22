param([ValidateSet('start','status','report','resume')][string]$Action='status')
$TaskRepo = Split-Path -Parent $PSScriptRoot
$TaskPreviousEncoding = $env:PYTHONIOENCODING
$TaskExitCode = 0
Push-Location -LiteralPath $TaskRepo
try {
    $env:PYTHONIOENCODING = 'utf-8'
    & "$TaskRepo\.venv\Scripts\python.exe" -m ff_prior_preserve323.job $Action
    $TaskExitCode = $LASTEXITCODE
} finally {
    $env:PYTHONIOENCODING = $TaskPreviousEncoding
    Pop-Location
}
exit $TaskExitCode
