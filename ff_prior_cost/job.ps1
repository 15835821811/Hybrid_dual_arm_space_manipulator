param(
    [ValidateSet('start','resume','status','report')][string]$Action = 'status',
    [string]$Output = 'artifacts/v7_1/equal_budget_residual_001'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repoRoot
try {
    & '.\.venv\Scripts\python.exe' -m ff_prior_cost.job $Action --output $Output
    exit $LASTEXITCODE
} finally { Pop-Location }
