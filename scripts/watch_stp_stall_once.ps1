param(
    [int]$StallMinutes = 25,
    [string]$RunName = "stp_public_method_v4",
    [string]$ProcessedName = "stp_public_method_v4"
)

$ErrorActionPreference = "SilentlyContinue"

$projectRoot = Split-Path -Parent $PSScriptRoot
$logRoot = "D:\Datasets\PPG_BP_Robustness\logs"
$runRoot = Join-Path $projectRoot ("outputs\{0}" -f $RunName)
$runLogRoot = Join-Path $logRoot $RunName
$completionFile = Join-Path $runRoot "F3\metrics.json"
$alertFile = Join-Path $runLogRoot "stp_stall_alert_once.json"
$monitorLog = Join-Path $runLogRoot "stp_stall_monitor_once.log"
$lastFingerprint = $null
$lastProgress = Get-Date

function Get-StpProcesses {
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match "run_stp_faithful_public.ps1|prepare_stp_public_data.py|train_stp.py|run_stp_faithful_ablations.ps1" -and
        ($_.CommandLine -match [regex]::Escape($RunName) -or $_.CommandLine -match [regex]::Escape($runRoot)) -and
        $_.CommandLine -notmatch "watch_stp_stall_once.ps1"
    })
}

function Get-Fingerprint {
    $latestLog = Get-ChildItem $runLogRoot -File |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $mimicCount = @(Get-ChildItem "D:\Datasets\PPG_BP_Robustness\processed\$ProcessedName\subjects\mimiciii\*_signals.npy").Count
    $latestState = Get-ChildItem $runRoot -Recurse -File |
        Where-Object { $_.Name -in @("best.pt", "last.pt", "history.json", "metrics.json") } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $processClock = (Get-StpProcesses | Measure-Object -Property UserModeTime -Sum).Sum
    $logIdentity = if ($latestLog) {
        "{0}|{1}|{2}" -f $latestLog.Name, $latestLog.Length, $latestLog.LastWriteTimeUtc.Ticks
    } else {
        "no-log"
    }
    $stateIdentity = if ($latestState) {
        "{0}|{1}|{2}" -f $latestState.FullName, $latestState.Length, $latestState.LastWriteTimeUtc.Ticks
    } else {
        "no-state"
    }
    "$logIdentity|$stateIdentity|mimic=$mimicCount|cpu=$processClock"
}

"$(Get-Date -Format o) monitor started; threshold=${StallMinutes}m" | Out-File $monitorLog -Append -Encoding UTF8
Remove-Item $alertFile -Force -ErrorAction SilentlyContinue

while ($true) {
    if (Test-Path $completionFile) {
        "$(Get-Date -Format o) completed normally; monitor exiting" | Out-File $monitorLog -Append -Encoding UTF8
        exit 0
    }

    $processes = Get-StpProcesses
    $fingerprint = Get-Fingerprint
    if ($fingerprint -ne $lastFingerprint) {
        $lastFingerprint = $fingerprint
        $lastProgress = Get-Date
    }

    $minutesWithoutProgress = ((Get-Date) - $lastProgress).TotalMinutes
    $reason = $null
    if ($processes.Count -eq 0) {
        $reason = "All STP pipeline processes disappeared before the summary was produced."
    } elseif ($minutesWithoutProgress -ge $StallMinutes) {
        $reason = "No log, data, checkpoint, or process-CPU change for at least $StallMinutes minutes."
    }

    if ($reason) {
        $latestLog = Get-ChildItem $runLogRoot -Filter "stp_faithful_*" -File |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        [pscustomobject]@{
            signal = 1
            detected_at = Get-Date -Format o
            reason = $reason
            last_progress_at = $lastProgress.ToString("o")
            latest_log = $latestLog.FullName
            latest_log_updated_at = $latestLog.LastWriteTime.ToString("o")
            process_ids = @($processes.ProcessId)
        } | ConvertTo-Json | Out-File $alertFile -Encoding UTF8
        "$(Get-Date -Format o) ALERT: $reason" | Out-File $monitorLog -Append -Encoding UTF8
        exit 1
    }

    Start-Sleep -Seconds 60
}
