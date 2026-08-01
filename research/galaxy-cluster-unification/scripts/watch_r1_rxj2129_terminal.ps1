param(
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot),
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$workspacePath = (Resolve-Path -LiteralPath $Workspace).Path
$stateDirectory = Join-Path $workspacePath "results\r1_rxj2129_terminal_watcher"
$statePath = Join-Path $stateDirectory "state.json"
$logPath = Join-Path $stateDirectory "watcher.log"
$h2Report = Join-Path $workspacePath "results\r1_rxj2129_hst_h2\report.json"
$x4Report = Join-Path $workspacePath "results\r1_rxj2129_xmm_x4_responses\report.json"
$terminalReport = Join-Path $workspacePath "results\r1_rxj2129_terminal_observable_disposition\report.json"
$goalReport = Join-Path $workspacePath "results\r0_r2_goal_progress\report.json"
$x4Marker = "\\wsl.localhost\Ubuntu-24.04\home\henry\.local\share\sigmagravity-xmm\work\rxj2129\0093030201\x4\cross_region_responses\.x4_response_products_complete"

New-Item -ItemType Directory -Force -Path $stateDirectory | Out-Null

function Write-WatcherState {
    param(
        [string]$Status,
        [string]$Message
    )
    $state = [ordered]@{
        watcher_version = "R1B3-RXJ2129-terminal-watcher-0.5"
        updated_utc = (Get-Date).ToUniversalTime().ToString("o")
        process_id = $PID
        status = $Status
        message = $Message
        h2_report_present = Test-Path -LiteralPath $h2Report
        x4_completion_marker_present = Test-Path -LiteralPath $x4Marker
        x4_report_present = Test-Path -LiteralPath $x4Report
        terminal_report_present = Test-Path -LiteralPath $terminalReport
    }
    $state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statePath -Encoding UTF8
    "{0} {1}: {2}" -f $state.updated_utc, $Status, $Message | Add-Content -LiteralPath $logPath -Encoding UTF8
}

Write-WatcherState -Status "watching" -Message "Waiting for immutable H2 report and X4 completion marker."

while ($true) {
    if ((Test-Path -LiteralPath $x4Marker) -and -not (Test-Path -LiteralPath $x4Report)) {
        Write-WatcherState -Status "auditing_x4" -Message "X4 marker exists; running the frozen response audit."
        & py -3.13 (Join-Path $workspacePath "scripts\audit_r1_rxj2129_xmm_x4_responses.py") *>> $logPath
        $auditExit = $LASTEXITCODE
        if (-not (Test-Path -LiteralPath $x4Report)) {
            Write-WatcherState -Status "x4_audit_error" -Message "X4 audit exited $auditExit without a report."
            exit 2
        }
    }

    if ((Test-Path -LiteralPath $h2Report) -and (Test-Path -LiteralPath $x4Report)) {
        Write-WatcherState -Status "finalizing" -Message "Both terminal component reports exist; running the frozen disposition."
        & py -3.13 (Join-Path $workspacePath "scripts\finalize_r1_rxj2129_terminal_observable_disposition.py") *>> $logPath
        $finalExit = $LASTEXITCODE
        if ($finalExit -eq 0 -and (Test-Path -LiteralPath $terminalReport)) {
            Write-WatcherState -Status "closing_goal_audit" -Message "Terminal disposition exists; running the full completion-evidence audit, master ledger, and closure tests."
            & py -3.13 (Join-Path $workspacePath "scripts\audit_r0_r2_completion_evidence.py") *>> $logPath
            $completionExit = $LASTEXITCODE
            & py -3.13 (Join-Path $workspacePath "scripts\audit_r0_r2_goal_progress.py") *>> $logPath
            $goalExit = $LASTEXITCODE
            & py -3.13 -m pytest -q `
                (Join-Path $workspacePath "tests\test_r1_rxj2129_terminal_disposition_protocol.py") `
                (Join-Path $workspacePath "tests\test_r1_rxj2129_terminal_disposition.py") `
                (Join-Path $workspacePath "tests\test_r1_rxj2129_hst_h2_execution.py") `
                (Join-Path $workspacePath "tests\test_r1_ten_system_public_data_ceiling.py") `
                (Join-Path $workspacePath "tests\test_r0_r2_completion_evidence.py") `
                (Join-Path $workspacePath "tests\test_r0_r2_goal_progress.py") *>> $logPath
            $testExit = $LASTEXITCODE
            if ($completionExit -eq 0 -and $goalExit -eq 0 -and $testExit -eq 0 -and (Test-Path -LiteralPath $goalReport)) {
                Write-WatcherState -Status "complete" -Message "Terminal RX J2129 disposition and master goal audit completed."
                exit 0
            }
            Write-WatcherState -Status "closure_audit_error" -Message "Completion evidence exited $completionExit, master audit exited $goalExit, and closure tests exited $testExit."
            exit 4
        }
        Write-WatcherState -Status "finalization_error" -Message "Terminal finalizer exited $finalExit."
        exit 3
    }

    Write-WatcherState -Status "watching" -Message "No terminal component pair yet."
    Start-Sleep -Seconds $PollSeconds
}
