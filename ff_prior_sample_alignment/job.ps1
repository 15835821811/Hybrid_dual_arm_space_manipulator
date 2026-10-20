param([ValidateSet('start','status','report','resume')][string]$Action='status')
$TaskRepo = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $TaskRepo
try { & "$TaskRepo\.venv\Scripts\python.exe" -m ff_prior_sample_alignment.job $Action }
finally { Pop-Location }
