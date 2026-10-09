$ErrorActionPreference = 'Stop'
$c3Archive = Split-Path -Parent $PSScriptRoot
$c3RunId = Split-Path -Leaf $c3Archive
$c3Events = @()
$c3ReadErrors = @()
$c3TeacherLog = Join-Path $PSScriptRoot 'teacher_search.log'
if (Test-Path -LiteralPath $c3TeacherLog) {
    foreach ($c3Line in Get-Content -LiteralPath $c3TeacherLog) {
        if ($c3Line -notlike '*C1_CANDIDATE_TERMINAL*') { continue }
        try { $c3Events += $c3Line | ConvertFrom-Json }
        catch { $c3ReadErrors += 'Incomplete teacher log line observed; recheck next snapshot.' }
    }
}
$c3Effects = @()
$c3TeacherRoot = Join-Path $c3Archive 'teacher_search'
if (Test-Path -LiteralPath $c3TeacherRoot) {
    foreach ($c3TaskDir in Get-ChildItem -LiteralPath $c3TeacherRoot -Directory) {
        foreach ($c3PairDir in Get-ChildItem -LiteralPath $c3TaskDir.FullName -Directory) {
            $c3EffectPath = Join-Path $c3PairDir.FullName 'teacher_effect.json'
            if (Test-Path -LiteralPath $c3EffectPath) { $c3Effects += $c3EffectPath }
        }
    }
}
$c3Receipts = @()
$c3ReceiptPaths = @(Get-ChildItem -LiteralPath $PSScriptRoot -Filter 'phase_*.json' -File)
$c3TeacherReceipt = Join-Path $c3Archive 'teacher_phase.json'
if (Test-Path -LiteralPath $c3TeacherReceipt) { $c3ReceiptPaths += Get-Item -LiteralPath $c3TeacherReceipt }
foreach ($c3ReceiptPath in $c3ReceiptPaths) {
    try {
        $c3Receipt = Get-Content -LiteralPath $c3ReceiptPath.FullName -Raw | ConvertFrom-Json
        $c3Receipts += [PSCustomObject]@{file=$c3ReceiptPath.Name; stage=$c3Receipt.stage; exit_code=$c3Receipt.exit_code; elapsed_wall_s=$c3Receipt.elapsed_wall_s}
    }
    catch { $c3ReadErrors += "Incomplete phase receipt observed: $($c3ReceiptPath.Name)" }
}
$c3Live = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.CommandLine -like '*search_aware_warmstart*' -and $_.CommandLine -like "*$c3RunId*"
} | ForEach-Object {
    $c3LiveStats = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
    [PSCustomObject]@{pid=$_.ProcessId; created=$_.CreationDate; cpu_seconds=$c3LiveStats.CPU; command=$_.CommandLine}
})
$c3FormalSnapshots = @()
foreach ($c3FormalStage in @('val', 'test')) {
    $c3SearchRelative = if ($c3FormalStage -eq 'val') { 'closed_loop_val/search' } else { 'test_search' }
    $c3ActualRelative = if ($c3FormalStage -eq 'val') { 'closed_loop_val/actual' } else { 'frozen_test/actual' }
    $c3SearchRoot = Join-Path $c3Archive $c3SearchRelative
    $c3FormalEvents = @()
    $c3CompletedRequests = @()
    $c3LatestFormal = $null
    $c3LatestWrite = [DateTime]::MinValue
    if (Test-Path -LiteralPath $c3SearchRoot) {
        foreach ($c3FormalTask in Get-ChildItem -LiteralPath $c3SearchRoot -Directory) {
            foreach ($c3FormalStream in Get-ChildItem -LiteralPath $c3FormalTask.FullName -Directory) {
                $c3FormalLog = Join-Path $c3FormalStream.FullName 'command.log'
                if (Test-Path -LiteralPath $c3FormalLog) {
                    $c3StreamEvents = @()
                    foreach ($c3FormalLine in Get-Content -LiteralPath $c3FormalLog) {
                        if ($c3FormalLine -notlike '*C1_CANDIDATE_TERMINAL*') { continue }
                        try { $c3StreamEvents += $c3FormalLine | ConvertFrom-Json }
                        catch { $c3ReadErrors += "Incomplete $c3FormalStage candidate line observed; recheck next snapshot." }
                    }
                    $c3FormalEvents += $c3StreamEvents
                    $c3LogInfo = Get-Item -LiteralPath $c3FormalLog
                    if ($c3StreamEvents.Count -gt 0 -and $c3LogInfo.LastWriteTimeUtc -gt $c3LatestWrite) {
                        $c3LatestWrite = $c3LogInfo.LastWriteTimeUtc
                        $c3LatestFormal = [PSCustomObject]@{task=$c3FormalTask.Name; stream=$c3FormalStream.Name; event=($c3StreamEvents | Select-Object -Last 1)}
                    }
                }
                $c3OuterPath = Join-Path $c3FormalStream.FullName 'outer_process.json'
                if (Test-Path -LiteralPath $c3OuterPath) {
                    try {
                        $c3Outer = Get-Content -LiteralPath $c3OuterPath -Raw | ConvertFrom-Json
                        $c3CompletedRequests += [PSCustomObject]@{task=$c3FormalTask.Name; stream=$c3FormalStream.Name; exit_code=$c3Outer.exit_code}
                    }
                    catch { $c3ReadErrors += "Incomplete $c3FormalStage outer receipt observed; recheck next snapshot." }
                }
            }
        }
    }
    $c3ActualSlots = @()
    $c3ActualRoot = Join-Path $c3Archive $c3ActualRelative
    if (Test-Path -LiteralPath $c3ActualRoot) {
        foreach ($c3ActualTask in Get-ChildItem -LiteralPath $c3ActualRoot -Directory) {
            foreach ($c3ActualEndpoint in Get-ChildItem -LiteralPath $c3ActualTask.FullName -Directory) {
                $c3ActualSlot = Join-Path $c3ActualEndpoint.FullName 'slot.json'
                if (Test-Path -LiteralPath $c3ActualSlot) { $c3ActualSlots += $c3ActualSlot }
            }
        }
    }
    $c3FormalSnapshots += [PSCustomObject]@{stage=$c3FormalStage; terminal_events=$c3FormalEvents.Count; completed_requests=$c3CompletedRequests; latest_event=$c3LatestFormal; actual_slot_files=$c3ActualSlots.Count}
}
[PSCustomObject]@{
    observed_utc=[DateTime]::UtcNow.ToString('o')
    teacher_terminal_events=$c3Events.Count
    teacher_completed_groups=$c3Effects.Count
    latest_teacher_event=($c3Events | Select-Object -Last 1)
    teacher_status_counts=@($c3Events | Group-Object status | Select-Object Name,Count)
    live_python_processes=$c3Live
    phase_receipts=$c3Receipts
    formal_stage_snapshots=$c3FormalSnapshots
    observation_read_errors=$c3ReadErrors
} | ConvertTo-Json -Depth 6
