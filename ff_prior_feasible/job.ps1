param(
    [ValidateSet('start','resume','status','report')][string]$Action = 'status',
    [string]$Output = 'artifacts/v7_1/goal_preserving_smoke_001'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repoRoot
try {
    & '.\.venv\Scripts\python.exe' -m ff_prior_feasible.job $Action --output $Output
    exit $LASTEXITCODE
} finally { Pop-Location }
