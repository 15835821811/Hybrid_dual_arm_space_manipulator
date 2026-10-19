param([ValidateSet('start','resume','status','report','await-source')][string]$Action = 'status')
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$previousEncoding = $env:PYTHONIOENCODING
Push-Location -LiteralPath $repoRoot
try {
    $env:PYTHONIOENCODING = 'utf-8'
    & '.\.venv\Scripts\python.exe' -m ff_prior_expand256_train.job $Action
    if ($LASTEXITCODE -ne 0) { throw "E256T job command failed: $LASTEXITCODE" }
} finally {
    $env:PYTHONIOENCODING = $previousEncoding
    Pop-Location
}
