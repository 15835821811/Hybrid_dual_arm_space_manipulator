param([ValidateSet('status','report','resume')][string]$Action='status')
$TaskPackage = Split-Path -Parent $PSScriptRoot
$TaskAnalysis = Join-Path $TaskPackage 'docs/ANALYSIS.md'
if ($Action -eq 'report' -and (Test-Path -LiteralPath $TaskAnalysis)) {
    Get-Content -LiteralPath $TaskAnalysis -Raw -Encoding UTF8
} else {
    & (Join-Path $TaskPackage 'job.ps1') $Action
}
