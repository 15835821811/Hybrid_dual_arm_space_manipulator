# Frozen experiment Python files are preserved. This entry point also handles
# numpy scalar serialization in the original auditor's returned dictionary.
param([string]$Output = 'artifacts/v7_1/latent_residual_smoke_001')
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$previousPythonEncoding = $env:PYTHONIOENCODING
$env:PYTHONIOENCODING = 'utf-8'
Push-Location -LiteralPath $repoRoot
try {
    & '.\.venv\Scripts\python.exe' -c 'import json,sys; from ff_prior_latent.audit_results import audit; from ff_prior.storage import plain; print(json.dumps(audit(sys.argv[1]),default=plain,ensure_ascii=False,indent=2))' $Output
    exit $LASTEXITCODE
} finally {
    Pop-Location
    $env:PYTHONIOENCODING = $previousPythonEncoding
}
