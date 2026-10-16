param([string]$Output='artifacts/v7_1/qualified_coverage_train_001')
$ErrorActionPreference='Stop'
$repoRoot=Split-Path -Parent $PSScriptRoot
$runDir=Join-Path $repoRoot $Output
$statePath=Join-Path $runDir 'status.json'
if (-not (Test-Path -LiteralPath $statePath)) {
    @{status='NOT_STARTED'; output=$runDir} | ConvertTo-Json
    exit 0
}
$state=Get-Content -LiteralPath $statePath -Raw -Encoding utf8 | ConvertFrom-Json
$alive=$null -ne (Get-Process -Id $state.pid -ErrorAction SilentlyContinue)
$counts=@{}
$inflight=$null
$ledgerPath=Join-Path $runDir 'ledger.json'
if (Test-Path -LiteralPath $ledgerPath) {
    $ledger=Get-Content -LiteralPath $ledgerPath -Raw -Encoding utf8 | ConvertFrom-Json
    foreach ($entry in $ledger.completed.PSObject.Properties.Value) {
        $key='{0}:{1}' -f $entry.stage,$entry.status
        if (-not $counts.ContainsKey($key)) { $counts[$key]=0 }
        $counts[$key]++
    }
    $inflight=$ledger.inflight
}
$updates=$null
$trainingPath=Join-Path $runDir 'learning/goal_loss/training.json'
if (Test-Path -LiteralPath $trainingPath) {
    $training=Get-Content -LiteralPath $trainingPath -Raw -Encoding utf8 | ConvertFrom-Json
    $updates=$training.updates
} elseif ($state.stage -eq 'train_augmented_diffusion') {
    $updates=$state.progress.updates
}
[ordered]@{
    status=$state.status
    stage=$state.stage
    worker_pid=$state.pid
    worker_alive=$alive
    elapsed_s=$state.elapsed_s
    heartbeat_age_s=[math]::Round(([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000)-$state.heartbeat,1)
    completed_by_stage_and_status=$counts
    inflight=$inflight
    training_updates=$updates
    last_worker_progress=$state.progress
    note='Read-only metadata snapshot; completed cache imports are historical computations, not new physics calls.'
} | ConvertTo-Json -Depth 6
