param(
    [ValidateSet('start','resume','status','report')][string]$Action = 'status',
    [string]$Output = 'artifacts/v7_1/paired_objectives_smoke_001'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$previousPythonEncoding = $env:PYTHONIOENCODING
$env:PYTHONIOENCODING = 'utf-8'
Push-Location -LiteralPath $repoRoot
try {
    $finalReport = Join-Path -Path $repoRoot -ChildPath 'ff_prior_paired_objectives/docs/ANALYSIS.md'
    if ($Action -eq 'report' -and $Output -eq 'artifacts/v7_1/paired_objectives_smoke_001' -and (Test-Path -LiteralPath $finalReport)) {
        Get-Content -LiteralPath $finalReport -Raw -Encoding utf8
        exit 0
    }
    & '.\.venv\Scripts\python.exe' -m ff_prior_paired_objectives.job $Action --output $Output
    exit $LASTEXITCODE
} finally {
    Pop-Location
    $env:PYTHONIOENCODING = $previousPythonEncoding
}
