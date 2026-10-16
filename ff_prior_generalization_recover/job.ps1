param(
    [ValidateSet('start','resume','status','report')][string]$Action = 'status',
    [string]$Output = 'artifacts/v7_1/independent_new_validation_recovery_001'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$previousPythonEncoding = $env:PYTHONIOENCODING
$env:PYTHONIOENCODING = 'utf-8'
Push-Location -LiteralPath $repoRoot
try {
    $finalReport = Join-Path -Path $repoRoot -ChildPath 'ff_prior_generalization/docs/ANALYSIS.md'
    if ($Action -eq 'report' -and $Output -eq 'artifacts/v7_1/independent_new_validation_recovery_001' -and (Test-Path -LiteralPath $finalReport)) {
        Get-Content -LiteralPath $finalReport -Raw -Encoding utf8
        exit 0
    }
    $analysisReport = Join-Path -Path $Output -ChildPath 'ANALYSIS.md'
    if ($Action -eq 'report' -and (Test-Path -LiteralPath $analysisReport)) {
        $reportText = Get-Content -LiteralPath $analysisReport -Raw -Encoding utf8
        $reportText.Replace('.\ff_prior_generalization\job.ps1', '.\ff_prior_generalization_recover\job.ps1')
        exit 0
    }
    & '.\.venv\Scripts\python.exe' -m ff_prior_generalization_recover.job $Action --output $Output
    exit $LASTEXITCODE
} finally {
    Pop-Location
    $env:PYTHONIOENCODING = $previousPythonEncoding
}
