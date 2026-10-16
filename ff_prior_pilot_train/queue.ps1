param([ValidateSet('start','status')][string]$Action='status')
$ErrorActionPreference='Stop'
Push-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
try {
    & '.\.venv\Scripts\python.exe' -m ff_prior_pilot_train.queue $Action
    exit $LASTEXITCODE
} finally { Pop-Location }
