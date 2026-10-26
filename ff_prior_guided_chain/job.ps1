param([ValidateSet('start','worker','resume','status','report')][string]$Action='status')
$taskRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $taskRoot
try { & .\.venv\Scripts\python.exe -m ff_prior_guided_chain.job $Action }
finally { Pop-Location }
