param([ValidateSet('status','report','resume')][string]$Action='status')
$TaskPackage = Split-Path -Parent $PSScriptRoot
$TaskAnalysis = Join-Path $TaskPackage 'docs/ANALYSIS.md'
if ($Action -eq 'report' -and (Test-Path -LiteralPath $TaskAnalysis)) {
    Get-Content -LiteralPath $TaskAnalysis -Raw -Encoding UTF8
} else {
    $TaskPreviousPythonEncoding = $env:PYTHONIOENCODING
    try {
        $env:PYTHONIOENCODING = 'utf-8'
        & (Join-Path $TaskPackage 'job.ps1') $Action
    } finally {
        $env:PYTHONIOENCODING = $TaskPreviousPythonEncoding
    }
}
