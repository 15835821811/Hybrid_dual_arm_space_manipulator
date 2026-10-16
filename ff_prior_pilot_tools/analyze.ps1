param([string]$Output='ff_prior_pilot_tools/evidence/current_snapshot')
$ErrorActionPreference='Stop'
$previousEncoding=$env:PYTHONIOENCODING
$env:PYTHONIOENCODING='utf-8'
Push-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
try {
    & '.\.venv\Scripts\python.exe' -m ff_prior_pilot_tools.analyze --output $Output
    exit $LASTEXITCODE
} finally { Pop-Location; $env:PYTHONIOENCODING=$previousEncoding }
