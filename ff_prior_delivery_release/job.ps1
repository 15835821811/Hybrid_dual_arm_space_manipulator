param([ValidateSet('status','report','check')][string]$Action = 'status')
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$previousEncoding = $env:PYTHONIOENCODING
Push-Location -LiteralPath $repoRoot
try {
    $env:PYTHONIOENCODING = 'utf-8'
    & '.\.venv\Scripts\python.exe' -m ff_prior_delivery_release.cli $Action
    if ($LASTEXITCODE -ne 0) { throw "Delivery command failed: $LASTEXITCODE" }
} finally {
    $env:PYTHONIOENCODING = $previousEncoding
    Pop-Location
}
