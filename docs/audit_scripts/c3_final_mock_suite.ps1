#Requires -Version 5.1
<#
.SYNOPSIS
Run final C.3 focused mock verification after the formal pipeline ends.
.DESCRIPTION
Authoring/static parsing is not execution. Fresh CIM must show no unapproved
external Python or formal pipeline; explicit loopback http.server PID exceptions
are verified by command shape and recorded. Original validate/report receipts must match exact
run/argv and exit zero. Creates only a NEW independent directory, never retries.
Builtin JUnit captures actual reported cases, not an assumed pass count.
Fresh CIM is a launch/end observation, not continuous workload surveillance.
.EXAMPLE
& 'E:\v64c3\docs\audit_scripts\c3_final_mock_suite.ps1' -OutputDirectory 'E:\v64c3\docs\audit_receipts\c3_final_mock_suite_01'
# Run only after formal validate/report and all Python jobs have exited.
# A rerun needs another NEW output directory. ExpectedCollectedCount is optional.
#>
[CmdletBinding()]
param(
    [string]$OutputDirectory = '',
    [ValidateRange(0, 2147483647)][int]$ExpectedCollectedCount = 0,
    [int[]]$AllowedLoopbackHttpServerPid = @()
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$RepoRoot = 'E:\v64c3'
$Python = 'E:\v64c2\.venv-c2\Scripts\python.exe'
$FormalRunRelative = 'v6_4/output/search_aware_warmstart_20261008_01'
$FormalRun = [IO.Path]::GetFullPath((Join-Path $RepoRoot $FormalRunRelative))
$Utf8 = New-Object Text.UTF8Encoding($false)
$TestFiles = @(
    'v6_4/tests/test_search_effect_teacher.py'
    'v6_4/tests/test_search_aware_models.py'
    'v6_4/tests/test_closed_loop_warmstart_validation.py'
    'v6_4/tests/test_search_aware_orchestrator.py'
    'v6_4/tests/test_route_initializers.py'
    'v6_4/tests/test_preference_diffusion_warmstart.py'
    'v6_4/tests/test_continuous_route_optimizer.py'
    'v6_4/tests/test_preference_warmstart_pipeline.py'
    'v6_4/tests/test_evaluate_planning.py'
    'v6_4/tests/test_residual_binding_current.py'
    'v6_4/tests/test_a1_geometry_evidence.py'
    'v6_4/tests/test_execution_aware_reference.py'
    'v6_4/tests/test_residual_execution.py'
    'v6_4/tests/test_search_aware_additional_contracts.py'
    'v6_4/visualization/test_report_search_aware_warmstart.py'
    'v6_4/visualization/test_search_aware_warmstart_media.py'
    'v6_4/visualization/test_export_search_aware_release.py'
)
# Exclude v6_4/test_preference_teacher_dataset.py: setUp hashes large archives.
# Bind key imported implementation and fixture text without collecting more tests.
# This explicit identity inventory is not a claim to cover every transitive import.
$BoundSources = @(
    'v6_4/search_effect_teacher.py', 'v6_4/simple_warmstart_regression.py',
    'v6_4/closed_loop_warmstart_validation.py', 'v6_4/search_aware_warmstart_experiment.py',
    'v6_4/route_initializers.py', 'v6_4/preference_diffusion_warmstart.py',
    'v6_4/continuous_route_optimizer.py', 'v6_4/route_optimizer_protocol.py',
    'v6_4/route_candidate_evaluator.py', 'v6_4/evaluate_preference_warmstart.py',
    'v6_4/evaluate_planning.py', 'v6_4/residual_execution.py', 'v6_4/reference_adapter.py',
    'v6_4/task_anchored_reference.py', 'v6_4/task_protocol.py', 'v6_4/residual_diffusion.py',
    'v6_4/proposal_gate.py', 'v6_4/conditional_execution.py', 'v6_4/preference_teacher_dataset.py',
    'v6_lite/run_v6_lite.py', 'v6_lite/hierarchical_qp.py', 'v6_lite/execution_ramp.py',
    'v6_lite/b2_interval_online.py', 'v6_lite/b2_interval_online_optimized.py',
    'v6_lite/b2_interval_runtime.py', 'model_test/robot_model_spec_v5.py',
    'model_test/whole_body_verifier_v5.py',
    'v6_4/visualization/report_search_aware_warmstart.py',
    'v6_4/visualization/build_search_aware_warmstart_media.py',
    'v6_4/visualization/export_search_aware_release.py',
    'v6_4/tests/test_execution_diagnostics.py', 'v6_4/tests/test_task_protocol.py'
)
function Utc-Now { [DateTime]::UtcNow.ToString('o') }
function Sha-File([string]$Path) {
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Write-NewText([string]$Path, [string]$Text) {
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
    try { $bytes = $Utf8.GetBytes($Text); $stream.Write($bytes, 0, $bytes.Length) }
    finally { $stream.Dispose() }
}
function Write-NewJson([string]$Path, $Value) {
    Write-NewText $Path (($Value | ConvertTo-Json -Depth 25) + [Environment]::NewLine)
}
function Is-Within([string]$Path, [string]$Parent) {
    $a = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    $b = [IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
    $a.Equals($b, [StringComparison]::OrdinalIgnoreCase) -or
        $a.StartsWith($b + '\', [StringComparison]::OrdinalIgnoreCase)
}
function Assert-NewOutput([string]$Path) {
    if (Test-Path -LiteralPath $Path) { throw "Output exists; choose a new directory: $Path" }
    foreach ($blocked in @($FormalRun, (Join-Path $RepoRoot 'v6_4/output'),
        (Join-Path $RepoRoot 'v6_4/releases'), (Join-Path $RepoRoot 'v6_lite/output'),
        (Join-Path $RepoRoot 'v6_lite/releases'))) {
        if (Is-Within $Path $blocked) { throw "Output must be outside formal output/releases: $Path" }
    }
    $cursor = [IO.Path]::GetDirectoryName($Path)
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if (-not $item.PSIsContainer) { throw "Output ancestor is not a directory: $cursor" }
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Output reparse-point ancestor unsupported: $cursor"
            }
            if ((Test-Path -LiteralPath (Join-Path $cursor 'release_manifest.json')) -or
                (Test-Path -LiteralPath (Join-Path $cursor 'portable_paths.json')) -or
                ((Test-Path -LiteralPath (Join-Path $cursor 'source_identity.json')) -and
                 (Test-Path -LiteralPath (Join-Path $cursor 'plan.json')))) {
                throw "Output ancestor is a release or sealed experiment: $cursor"
            }
        }
        $next = [IO.Path]::GetDirectoryName($cursor)
        if ($next -eq $cursor) { break }; $cursor = $next
    }
}
function Fresh-ProcessCheck([string]$Purpose) {
    # A CIM query failure is an error, never an empty process list.
    $all = @(Get-CimInstance -ClassName Win32_Process -ErrorAction Stop)
    $blockers = @(); $backgroundServers = @()
    foreach ($process in $all) {
        if ([int]$process.ProcessId -eq $PID) { continue }
        $name = [string]$process.Name; $command = [string]$process.CommandLine
        $isPython = $name -match '(?i)^(python|pypy).*\.exe$'
        $isFormal = (($command -match 'search_aware_warmstart_20261008_01') -and
            ($command -match '(?i)continue_pipeline\.ps1|v6_4[./\\]search_aware_warmstart_experiment|teacher-search|closed-loop-val|test-search|execute-test|run-all'))
        if ($isPython -and -not $isFormal -and
            ($AllowedLoopbackHttpServerPid -contains [int]$process.ProcessId) -and
            ($command -match '\s-m\s+http\.server\s+[0-9]+\s+--bind\s+127\.0\.0\.1\s*$')) {
            $backgroundServers += [ordered]@{pid = [int]$process.ProcessId; name = $name
                created = $process.CreationDate; executable_path = [string]$process.ExecutablePath
                cpu_seconds = ([double]$process.KernelModeTime + [double]$process.UserModeTime) / 10000000
                reason = 'explicit PID; verified Python standard-library HTTP server bound only to loopback'}
            continue
        }
        if ($isPython -or $isFormal) {
            # Arbitrary command lines may contain secrets; do not persist them.
            $blockers += [ordered]@{pid = [int]$process.ProcessId; name = $name
                executable_path = [string]$process.ExecutablePath
                python_process = $isPython; formal_pipeline_process = $isFormal}
        }
    }
    [ordered]@{utc = Utc-Now; purpose = $Purpose; cim_process_count = $all.Count
        blockers = $blockers; allowed_background_servers = $backgroundServers}
}
function Assert-Quiet($Check) {
    if (@($Check.blockers).Count -gt 0) {
        $labels = @($Check.blockers | ForEach-Object { "$($_.name) PID=$($_.pid)" })
        throw ("Refusing parallel execution: " + ($labels -join ', '))
    }
}
function Read-StageReceipt([string]$Stage) {
    $path = Join-Path $FormalRun ("command_logs/phase_" + $Stage + '.json')
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Formal receipt missing: $path" }
    $item = Get-Item -LiteralPath $path
    if ($item.Length -gt 1048576) { throw "Oversized formal receipt: $path" }
    $raw = [IO.File]::ReadAllText($path, $Utf8)
    if ((Get-Command ConvertFrom-Json).Parameters.ContainsKey('DateKind')) {
        $value = $raw | ConvertFrom-Json -DateKind String
    } else {
        $value = $raw | ConvertFrom-Json
        foreach ($dateKey in @('started_utc', 'ended_utc')) {
            if ($value.$dateKey -is [DateTime]) { $value.$dateKey = $value.$dateKey.ToString('o') }
        }
    }
    foreach ($key in @('stage', 'argv', 'started_utc', 'ended_utc', 'exit_code', 'automatic_retry')) {
        if ($value.PSObject.Properties.Name -notcontains $key) { throw "Receipt lacks $key : $path" }
    }
    $integerExit = ($value.exit_code -is [int]) -or ($value.exit_code -is [long])
    if ($value.stage -cne $Stage -or -not $integerExit -or $value.exit_code -ne 0 -or
        $value.automatic_retry -isnot [bool] -or $value.automatic_retry) {
        throw "Receipt is not original exit-zero non-retry completion: $path"
    }
    $expected = @($Python, '-u', '-B', '-X', 'utf8', '-m',
        'v6_4.search_aware_warmstart_experiment', $Stage, '--run', $FormalRunRelative)
    $actual = @($value.argv)
    if ($actual.Count -ne $expected.Count) { throw "Unexpected receipt argv: $path" }
    for ($i = 0; $i -lt $expected.Count; $i++) {
        if ([string]$actual[$i] -cne $expected[$i]) { throw "Wrong run/interpreter/stage argv: $path" }
    }
    $started = [DateTimeOffset]::Parse([string]$value.started_utc)
    $ended = [DateTimeOffset]::Parse([string]$value.ended_utc)
    if ($started.Offset -ne [TimeSpan]::Zero -or $ended.Offset -ne [TimeSpan]::Zero -or
        $ended -lt $started -or $ended -gt [DateTimeOffset]::UtcNow.AddMinutes(1)) {
        throw "Invalid original UTC interval: $path"
    }
    [ordered]@{path = $path; bytes = $item.Length; sha256 = Sha-File $path; content = $value}
}
function Test-Inventory {
    foreach ($relative in $TestFiles) {
        $path = Join-Path $RepoRoot $relative; $item = Get-Item -LiteralPath $path
        if ($item.PSIsContainer -or $item.Length -gt 1048576) { throw "Unexpected test source: $path" }
        [pscustomobject][ordered]@{path = $relative; absolute_path = $item.FullName; bytes = $item.Length
            sha256 = Sha-File $path
            static_test_functions = @(Select-String -LiteralPath $path -Pattern '^\s*(?:async\s+)?def test_').Count}
    }
}
function Bound-SourceInventory {
    foreach ($relative in $BoundSources) {
        $path = Join-Path $RepoRoot $relative; $item = Get-Item -LiteralPath $path
        if ($item.PSIsContainer -or $item.Length -gt 1048576) { throw "Unexpected bound text source: $path" }
        [pscustomobject][ordered]@{path = $relative; bytes = $item.Length; sha256 = Sha-File $path}
    }
}
function Git-Head {
    $result = @(& git -C $RepoRoot rev-parse --verify HEAD 2>&1)
    if ($LASTEXITCODE -ne 0 -or $result.Count -ne 1 -or [string]$result[0] -notmatch '^[0-9a-fA-F]{40}$') {
        throw 'Cannot record Git HEAD.'
    }
    ([string]$result[0]).ToLowerInvariant()
}
function Windows-Argument([string]$Value) {
    # Windows CRT quoting; no cmd.exe or PowerShell interpolation is involved.
    '"' + [regex]::Replace([regex]::Replace($Value, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}
function Invoke-Captured([string[]]$Arguments, [string]$Prefix, $Overrides) {
    $outPath = Join-Path $OutputDirectory ($Prefix + '.stdout.txt')
    $errPath = Join-Path $OutputDirectory ($Prefix + '.stderr.txt')
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $Python; $info.WorkingDirectory = $RepoRoot
    $info.UseShellExecute = $false; $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true; $info.RedirectStandardError = $true
    if ($info.PSObject.Properties.Name -contains 'ArgumentList') {
        foreach ($arg in $Arguments) { $info.ArgumentList.Add($arg) }
    } else { $info.Arguments = (@($Arguments | ForEach-Object { Windows-Argument $_ }) -join ' ') }
    foreach ($entry in $Overrides.GetEnumerator()) { $info.EnvironmentVariables[$entry.Key] = [string]$entry.Value }
    $outFile = [IO.File]::Open($outPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
    $errFile = $null; $process = New-Object Diagnostics.Process; $process.StartInfo = $info
    $started = Utc-Now; $clock = [Diagnostics.Stopwatch]::StartNew()
    # Retain launch/PID/native exit even if an output-copy/finalization step fails.
    $invocation = [ordered]@{argv = @($Python) + @($Arguments); cwd = $RepoRoot
        started_utc = $started; ended_utc = $null; elapsed_wall_s = $null
        launched = $false; pid = $null; exit_code = $null
        stdout_path = $outPath; stderr_path = $errPath; output_capture_completed = $false}
    if ($Prefix -eq 'pytest') { $receipt.pytest = $invocation }
    else { $receipt.interpreter_probe = $invocation }
    try {
        $errFile = [IO.File]::Open($errPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
        if (-not $process.Start()) { throw "Failed to launch $Prefix" }
        $invocation.launched = $true; $invocation.pid = $process.Id
        # Concurrent persistent stream copies avoid pipe deadlocks and retain partial output.
        $outCopy = $process.StandardOutput.BaseStream.CopyToAsync($outFile)
        $errCopy = $process.StandardError.BaseStream.CopyToAsync($errFile)
        $process.WaitForExit(); $invocation.exit_code = $process.ExitCode
        [void]$outCopy.GetAwaiter().GetResult(); [void]$errCopy.GetAwaiter().GetResult()
        $outFile.Flush(); $errFile.Flush()
        $invocation.output_capture_completed = $true
        $invocation
    } catch {
        $invocation.capture_or_launch_error = $_.Exception.Message
        throw
    } finally {
        if ($invocation.launched -and $null -eq $invocation.exit_code) {
            try { if ($process.HasExited) { $invocation.exit_code = $process.ExitCode } } catch { }
        }
        $invocation.ended_utc = Utc-Now; $invocation.elapsed_wall_s = $clock.Elapsed.TotalSeconds
        $clock.Stop(); $outFile.Dispose(); if ($null -ne $errFile) { $errFile.Dispose() }; $process.Dispose()
    }
}

$receipt = [ordered]@{
    schema = 'v64_c3_final_focused_mock_suite_receipt_v1'
    wrapper_started_utc = Utc-Now; repo = $RepoRoot; formal_run = $FormalRun
    expected_collected_count = $(if ($ExpectedCollectedCount -gt 0) { $ExpectedCollectedCount } else { $null })
    role = 'post-freeze software verification; not a new scientific experiment'
    automatic_retry = $false; status = 'PREFLIGHT'; process_checks = @(); formal_stage_receipts = @()
    allowed_loopback_http_server_pids = $AllowedLoopbackHttpServerPid
    pytest = $null; interpreter_probe = $null
    limitations = @(
        'Fresh CIM observations are launch/end checks, not continuous workload surveillance.'
        'A mock pass does not certify experiment completion, physics or independent gates.'
        'JUnit describes reported cases; static function counts never imply collection or passes.'
        'Explicit implementation/fixture hashes bind selected source text, not all transitive imports or installed packages.'
    )
}
$outputCreated = $false; $receiptWritten = $false; $wrapperExit = 2
try {
    $check = Fresh-ProcessCheck 'before any interpreter or inventory work'
    $receipt.process_checks += $check; Assert-Quiet $check
    $validate = Read-StageReceipt 'validate'; $report = Read-StageReceipt 'report'
    if ([DateTimeOffset]::Parse($report.content.started_utc) -lt [DateTimeOffset]::Parse($validate.content.ended_utc)) {
        throw 'Report starts before validate completion.'
    }
    $receipt.formal_stage_receipts = @($validate, $report)
    if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
        $name = 'c3_final_mock_suite_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '_' + [Guid]::NewGuid().ToString('N')
        $OutputDirectory = Join-Path $RepoRoot ('docs/audit_receipts/' + $name)
    }
    $OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory); Assert-NewOutput $OutputDirectory
    $null = New-Item -ItemType Directory -Path $OutputDirectory; $outputCreated = $true
    $tmp = Join-Path $OutputDirectory 'tmp'; $null = New-Item -ItemType Directory -Path $tmp
    $receipt.output_directory = $OutputDirectory; $receipt.head_before = Git-Head
    $receipt.test_sources_before = @(Test-Inventory); $receipt.test_module_count = $receipt.test_sources_before.Count
    $receipt.bound_sources_before = @(Bound-SourceInventory)
    $receipt.static_source_function_count = ($receipt.test_sources_before | Measure-Object static_test_functions -Sum).Sum
    $exe = Get-Item -LiteralPath $Python
    $receipt.interpreter_file = [ordered]@{path = $exe.FullName; bytes = $exe.Length; sha256 = Sha-File $Python
        pe_file_version = $exe.VersionInfo.FileVersion; pe_product_version = $exe.VersionInfo.ProductVersion}
    $venvConfig = Join-Path ([IO.Directory]::GetParent($exe.DirectoryName).FullName) 'pyvenv.cfg'
    if (Test-Path -LiteralPath $venvConfig -PathType Leaf) {
        $receipt.interpreter_file.pyvenv_config = [ordered]@{path = $venvConfig; sha256 = Sha-File $venvConfig
            content = [IO.File]::ReadAllText($venvConfig, $Utf8)}
    }
    $receipt.wrapper_source = [ordered]@{path = $PSCommandPath; bytes = (Get-Item -LiteralPath $PSCommandPath).Length; sha256 = Sha-File $PSCommandPath}
    $childEnv = [ordered]@{
        PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'; PYTEST_ADDOPTS = ''; PYTEST_PLUGINS = ''
        OMP_NUM_THREADS = '1'; OPENBLAS_NUM_THREADS = '1'; MKL_NUM_THREADS = '1'; NUMBA_NUM_THREADS = '1'
        CUDA_VISIBLE_DEVICES = ''; TEMP = $tmp; TMP = $tmp; PYTEST_DEBUG_TEMPROOT = $tmp
    }
    $receipt.child_environment_overrides = $childEnv
    $receipt.environment_note = 'Other parent environment is inherited. ProcessStartInfo supplies empty CUDA explicitly; ADDOPTS/PLUGINS cleared for the explicit suite.'
    $junitPath = Join-Path $OutputDirectory 'pytest.junit.xml'
    $pytestArgv = @('-B', '-X', 'utf8', '-m', 'pytest', '--import-mode=importlib',
        '-p', 'no:cacheprovider', '-q') + $TestFiles + @('--junitxml=' + $junitPath)
    $receipt.planned_pytest_argv = @($Python) + $pytestArgv; $receipt.status = 'STARTED'
    Write-NewJson (Join-Path $OutputDirectory 'started.json') $receipt

    $check = Fresh-ProcessCheck 'immediately before stdlib interpreter identity probe'
    $receipt.process_checks += $check; Assert-Quiet $check
    $probeCode = 'import json,sys,struct; print(json.dumps(dict(executable=sys.executable,version=sys.version,version_info=list(sys.version_info),implementation=sys.implementation.name,prefix=sys.prefix,base_prefix=sys.base_prefix,pointer_bits=struct.calcsize("P")*8,dont_write_bytecode=sys.dont_write_bytecode),sort_keys=True))'
    $probe = Invoke-Captured @('-B', '-X', 'utf8', '-c', $probeCode) 'interpreter' $childEnv
    $receipt.interpreter_probe = $probe
    if ($probe.exit_code -ne 0) { throw 'Interpreter identity probe failed; pytest not launched.' }
    $receipt.interpreter_runtime = [IO.File]::ReadAllText($probe.stdout_path, $Utf8) | ConvertFrom-Json
    if (-not ([IO.Path]::GetFullPath($receipt.interpreter_runtime.executable).Equals(
        [IO.Path]::GetFullPath($Python), [StringComparison]::OrdinalIgnoreCase))) { throw 'Wrong runtime executable.' }

    $check = Fresh-ProcessCheck 'immediately before the single pytest invocation'
    $receipt.process_checks += $check; Assert-Quiet $check
    $suite = Invoke-Captured $pytestArgv 'pytest' $childEnv; $receipt.pytest = $suite
    $receipt.head_after = Git-Head; $receipt.test_sources_after = @(Test-Inventory)
    $receipt.bound_sources_after = @(Bound-SourceInventory)
    $receipt.source_identity_unchanged = (($receipt.test_sources_before | ConvertTo-Json -Depth 5 -Compress) -ceq
        ($receipt.test_sources_after | ConvertTo-Json -Depth 5 -Compress)) -and ($receipt.head_before -ceq $receipt.head_after) -and
        (($receipt.bound_sources_before | ConvertTo-Json -Depth 5 -Compress) -ceq
         ($receipt.bound_sources_after | ConvertTo-Json -Depth 5 -Compress)) -and
        ((Sha-File $PSCommandPath) -ceq $receipt.wrapper_source.sha256)
    $receipt.formal_receipts_unchanged = ((Sha-File $validate.path) -ceq $validate.sha256) -and ((Sha-File $report.path) -ceq $report.sha256)
    $receipt.interpreter_unchanged = (Sha-File $Python) -ceq $receipt.interpreter_file.sha256
    $check = Fresh-ProcessCheck 'after pytest exited'; $receipt.process_checks += $check
    $receipt.no_unapproved_external_process_at_end = (@($check.blockers).Count -eq 0)
    foreach ($record in @($probe, $suite)) {
        $record.stdout_sha256 = Sha-File $record.stdout_path; $record.stderr_sha256 = Sha-File $record.stderr_path
    }
    if (-not (Test-Path -LiteralPath $junitPath -PathType Leaf)) { throw 'Original pytest JUnit missing; receipt incomplete.' }
    $xml = New-Object Xml.XmlDocument; $xml.XmlResolver = $null; $xml.Load($junitPath)
    $suites = @($xml.SelectNodes('/testsuites/testsuite'))
    if ($suites.Count -eq 0) { throw 'JUnit contains no suites.' }
    $counts = [ordered]@{reported_cases = 0; failures = 0; errors = 0; skipped = 0}
    foreach ($entry in $suites) {
        $counts.reported_cases += [int]$entry.GetAttribute('tests'); $counts.failures += [int]$entry.GetAttribute('failures')
        $counts.errors += [int]$entry.GetAttribute('errors'); $counts.skipped += [int]$entry.GetAttribute('skipped')
    }
    $counts.passed_cases = $counts.reported_cases - $counts.failures - $counts.errors - $counts.skipped
    $receipt.junit = [ordered]@{path = $junitPath; sha256 = Sha-File $junitPath; counts = $counts}
    $wrapperExit = [int]$suite.exit_code
    $receipt.status = $(if ($wrapperExit -eq 0) { 'PYTEST_EXIT_ZERO' } else { 'PYTEST_NONZERO' })
    if (-not $receipt.source_identity_unchanged -or -not $receipt.formal_receipts_unchanged -or
        -not $receipt.interpreter_unchanged -or -not $receipt.no_unapproved_external_process_at_end) {
        $wrapperExit = 3; $receipt.status = 'AUDIT_IDENTITY_OR_CONCURRENCY_FAILURE'
    } elseif ($suite.exit_code -eq 0 -and ($counts.reported_cases -eq 0 -or
        ($ExpectedCollectedCount -gt 0 -and $counts.reported_cases -ne $ExpectedCollectedCount))) {
        $wrapperExit = 4; $receipt.status = 'ACTUAL_CASE_COUNT_MISMATCH'
    }
} catch {
    $receipt.status = $(if ($null -eq $receipt.pytest -or -not $receipt.pytest.launched) {
        'REFUSED_OR_PRE_PYTEST_FAILURE'
    } else { 'PYTEST_OUTPUT_OR_POST_RUN_RECEIPT_FAILURE' })
    $receipt.wrapper_error = [ordered]@{type = $_.Exception.GetType().FullName; message = $_.Exception.Message}
    $wrapperExit = 2; [Console]::Error.WriteLine($_.Exception.Message)
} finally {
    $receipt.wrapper_ended_utc = Utc-Now; $receipt.wrapper_exit_code = $wrapperExit
    if ($outputCreated) {
        try {
            $md = @('# C.3 final focused mock-suite receipt', '',
                ('Status: ' + $receipt.status + '; wrapper exit: ' + $wrapperExit + '.'),
                ('UTC: ' + $receipt.wrapper_started_utc + ' to ' + $receipt.wrapper_ended_utc + '.'),
                ('Exact argv/env, identity and original output: receipt.json in ' + $OutputDirectory + '.'),
                '', 'Post-freeze software verification; static function counts are not pass counts.',
                'Native pytest exit and JUnit reported cases are recorded separately in JSON.',
                'Launch/end CIM checks do not establish continuous workload isolation.',
                'A mock pass does not substitute for the original experiment or independent gates.') -join [Environment]::NewLine
            Write-NewText (Join-Path $OutputDirectory 'RECEIPT.md') ($md + [Environment]::NewLine)
        } catch {
            [Console]::Error.WriteLine('Markdown receipt failed: ' + $_.Exception.Message)
            $receipt.ancillary_receipt_error = $_.Exception.Message
            $receipt.status = 'RECEIPT_FINALIZATION_FAILURE'; $wrapperExit = 2
        }
        $receipt.wrapper_exit_code = $wrapperExit
        try {
            Write-NewJson (Join-Path $OutputDirectory 'receipt.json') $receipt
            $receiptWritten = $true
        } catch {
            [Console]::Error.WriteLine('JSON receipt finalization failed: ' + $_.Exception.Message)
            $wrapperExit = 2
        }
    }
}
if ($receiptWritten) { Write-Output ('Independent audit receipt: ' + (Join-Path $OutputDirectory 'receipt.json')) }
exit $wrapperExit

