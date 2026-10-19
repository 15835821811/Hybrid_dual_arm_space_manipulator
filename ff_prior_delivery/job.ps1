param([ValidateSet('status','plan','report')][string]$Action = 'status')
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if ($Action -eq 'plan') {
    Get-Content -LiteralPath (Join-Path $repoRoot 'paper/EXPERIMENT_PLAN.md') -Raw -Encoding UTF8
} elseif ($Action -eq 'report') {
    Get-Content -LiteralPath (Join-Path $PSScriptRoot 'docs/ANALYSIS.md') -Raw -Encoding UTF8
} else {
    Push-Location -LiteralPath $repoRoot
    $previousEncoding = $env:PYTHONIOENCODING
    try {
        $env:PYTHONIOENCODING = 'utf-8'
        $raw = & '.\.venv\Scripts\python.exe' -m ff_prior_delivery.status
        if ($LASTEXITCODE -ne 0) { throw 'Status reader failed' }
        $result = ($raw -join [Environment]::NewLine) | ConvertFrom-Json
        foreach ($entry in $result.stages) {
            if ($entry.PSObject.Properties.Name -contains 'runtime') {
                if ($entry.runtime.PSObject.Properties.Name -contains 'pid') {
                    $process = Get-Process -Id $entry.runtime.pid -ErrorAction SilentlyContinue
                    $entry.runtime | Add-Member -NotePropertyName pid_alive -NotePropertyValue ([bool]$process)
                }
            }
        }
        $result | ConvertTo-Json -Depth 12
    } finally {
        $env:PYTHONIOENCODING = $previousEncoding
        Pop-Location
    }
}
