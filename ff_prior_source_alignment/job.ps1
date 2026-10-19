param([ValidateSet('status','report','plan')][string]$Action='status')
$ErrorActionPreference = 'Stop'
if ($Action -eq 'report') {
    Get-Content -LiteralPath (Join-Path $PSScriptRoot 'docs/ANALYSIS.md') -Raw -Encoding UTF8
} elseif ($Action -eq 'plan') {
    Get-Content -LiteralPath (Join-Path $PSScriptRoot 'docs/NEXT_PROTOCOL.md') -Raw -Encoding UTF8
} else {
    $inventory = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'evidence/local_alignment_002.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    [ordered]@{
        source_audit='COMPLETE'
        current_data_inventory=$inventory.status
        prospective_D_TRAIN_gate=$inventory.prospective_metrics.D_both.TRAIN_endpoint_gate
        next_protocol='DESIGNED_NOT_IMPLEMENTED_OR_LAUNCHED'
        new_training=$false
        new_DEV=$false
        TEST=$false
    } | ConvertTo-Json
}
