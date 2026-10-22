param([ValidateSet('start','status','report','resume')][string]$Action='status')
$TaskRepo = Split-Path -Parent $PSScriptRoot
$TaskPreviousEncoding = $env:PYTHONIOENCODING
Push-Location -LiteralPath $TaskRepo
try {
    $env:PYTHONIOENCODING = 'utf-8'
    & "$TaskRepo\.venv\Scripts\python.exe" -m ff_prior_static_fullchain.job $Action
} finally {
    $env:PYTHONIOENCODING = $TaskPreviousEncoding
    Pop-Location
}
