param(
    [ValidateSet('start','resume','status','report')][string]$Action = 'status',
    [string]$Output = 'artifacts/v7_1/expansion_boundary_audit_001'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$previousPythonEncoding = $env:PYTHONIOENCODING
$env:PYTHONIOENCODING = 'utf-8'
Push-Location -LiteralPath $repoRoot
try {
    $analysisReport = Join-Path -Path $Output -ChildPath 'ANALYSIS.md'
    if ($Action -eq 'report' -and (Test-Path -LiteralPath $analysisReport)) {
        Get-Content -LiteralPath $analysisReport -Raw -Encoding utf8
        exit 0
    }
    & '.\.venv\Scripts\python.exe' -m ff_prior_boundary.job $Action --output $Output
    exit $LASTEXITCODE
} finally {
    Pop-Location
    $env:PYTHONIOENCODING = $previousPythonEncoding
}
