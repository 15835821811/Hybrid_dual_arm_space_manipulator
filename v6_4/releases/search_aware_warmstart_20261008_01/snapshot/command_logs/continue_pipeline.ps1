$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'E:\v64c3'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
$c3Run = 'v6_4/output/search_aware_warmstart_20261008_01'
$c3Python = 'E:\v64c2\.venv-c2\Scripts\python.exe'
while (-not (Test-Path -LiteralPath "$c3Run/teacher_phase.json")) {
    Start-Sleep -Seconds 5
}
$c3Teacher = Get-Content -LiteralPath "$c3Run/teacher_phase.json" -Raw | ConvertFrom-Json
if ($c3Teacher.exit_code -ne 0) { throw 'Teacher command failed; no automatic retry.' }
foreach ($c3Stage in @('build-dataset', 'train', 'closed-loop-val', 'freeze-models', 'test-search', 'execute-test', 'validate', 'report')) {
    $c3Receipt = "$c3Run/command_logs/phase_$c3Stage.json"
    if (Test-Path -LiteralPath $c3Receipt) { throw "Existing stage receipt requires explicit verification: $c3Stage" }
    $c3Started = [DateTime]::UtcNow.ToString('o')
    $c3Clock = [Diagnostics.Stopwatch]::StartNew()
    $c3Argv = @('-u', '-B', '-X', 'utf8', '-m', 'v6_4.search_aware_warmstart_experiment', $c3Stage, '--run', $c3Run)
    Write-Output "C3_STAGE_START $c3Stage $c3Started"
    & $c3Python @c3Argv 2>&1 | Tee-Object -FilePath "$c3Run/command_logs/$c3Stage.log"
    $c3Exit = $LASTEXITCODE
    @{stage=$c3Stage;argv=@($c3Python)+$c3Argv;started_utc=$c3Started;ended_utc=[DateTime]::UtcNow.ToString('o');elapsed_wall_s=$c3Clock.Elapsed.TotalSeconds;exit_code=$c3Exit;automatic_retry=$false} | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 -LiteralPath $c3Receipt
    if ($c3Exit -ne 0) { throw "Consumed stage failed, retained without retry: $c3Stage" }
}
