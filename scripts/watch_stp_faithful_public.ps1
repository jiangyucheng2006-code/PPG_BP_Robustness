param(
    [string]$RunName = "stp_public_method_v4",
    [string]$ProcessedName = "stp_public_method_v4"
)

$ErrorActionPreference = "SilentlyContinue"

$logRoot = "D:\Datasets\PPG_BP_Robustness\logs"
$processedRoot = "D:\Datasets\PPG_BP_Robustness\processed\$ProcessedName"
$runRoot = Join-Path (Split-Path -Parent $PSScriptRoot) ("outputs\{0}" -f $RunName)
$runLogRoot = Join-Path $logRoot $RunName
$host.UI.RawUI.WindowTitle = "STP Public Method - Live Progress"

while ($true) {
    $pipeline = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match "run_stp_faithful_public.ps1|run_stp_faithful_ablations.ps1" -and
        $_.CommandLine -match [regex]::Escape($RunName) -and
        $_.CommandLine -notmatch "watch_stp_faithful_public"
    } | Select-Object -First 1
    $worker = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match "prepare_stp_public_data.py|train_stp.py" -and
        ($_.CommandLine -match [regex]::Escape($runRoot) -or $_.CommandLine -match [regex]::Escape($RunName))
    } | Select-Object -First 1

    $latestLog = Get-ChildItem $runLogRoot -File |
        Where-Object {
            $_.Name -like "stp_faithful_*.log" -or
            $_.Name -in @("pipeline.out.log", "pipeline.err.log")
        } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1

    $stage = "Waiting"
    if ($worker.CommandLine -match "prepare_stp_public_data.py") {
        $stage = "DATA PREPARATION"
    } elseif ($worker.CommandLine -match "_ablations") {
        $outputMatch = [regex]::Match($worker.CommandLine, "--output\s+(\S+)")
        $experimentName = if ($outputMatch.Success) {
            Split-Path $outputMatch.Groups[1].Value -Leaf
        } else {
            "comparison"
        }
        $stage = "OVERNIGHT ABLATION - $experimentName"
    } elseif ($worker.CommandLine -match "[\\/]F1(?:[\\/]|\s|$)") {
        $stage = "F1 - SELF-SUPERVISED PRETRAINING"
    } elseif ($worker.CommandLine -match "[\\/]F2(?:[\\/]|\s|$)") {
        $stage = "F2 - BP PATTERN ADAPTATION"
    } elseif ($worker.CommandLine -match "[\\/]F3(?:[\\/]|\s|$)") {
        $stage = "F3 - BP REGRESSION"
    }

    $wesad = @(Get-ChildItem "$processedRoot\subjects\wesad\*_signals.npy").Count
    $dalia = @(Get-ChildItem "$processedRoot\subjects\ppgdalia\*_signals.npy").Count
    $mimic = @(Get-ChildItem "$processedRoot\subjects\mimiciii\*_signals.npy").Count
    $elapsed = if ($pipeline) {
        (Get-Date) - [datetime]$pipeline.CreationDate
    } else {
        [TimeSpan]::Zero
    }
    $gpu = & nvidia-smi `
        --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu `
        --format=csv,noheader,nounits 2>$null
    $stageHistory = $null
    foreach ($stageName in @("F1", "F2", "F3")) {
        $historyPath = Join-Path $runRoot "$stageName\history.json"
        if (Test-Path $historyPath) {
            try {
                $rows = Get-Content $historyPath -Raw | ConvertFrom-Json
                if ($rows.Count -gt 0) {
                    $lastRow = $rows[-1]
                    $stageHistory = "$stageName completed epoch $($lastRow.epoch)"
                }
            } catch {}
        }
    }

    Clear-Host
    Write-Host ("STP public-method reproduction: {0}" -f $RunName) -ForegroundColor Cyan
    Write-Host ("Updated: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    Write-Host ("Stage:   {0}" -f $stage) -ForegroundColor Yellow
    Write-Host ("Status:  {0}" -f $(if ($pipeline) { "RUNNING" } else { "STOPPED / COMPLETE" }))
    Write-Host ("Elapsed: {0:hh\:mm\:ss}" -f $elapsed)
    Write-Host ("Subjects prepared: WESAD {0}/15 | PPG-DaLiA {1}/15 | MIMIC files {2}" -f $wesad, $dalia, $mimic)
    if ($stageHistory) { Write-Host ("Epoch status: {0}" -f $stageHistory) }
    if ($gpu) { Write-Host ("GPU: {0}% | VRAM {1}/{2} MiB | {3} C" -f (($gpu -split ',').Trim())) }
    Write-Host "Initial total estimate: about 8-13 hours; recalibrated after F1 epoch 1."
    Write-Host ""
    Write-Host "Latest progress" -ForegroundColor Green
    if ($latestLog) {
        Get-Content $latestLog.FullName -Tail 20
    }
    Write-Host ""
    Write-Host "Closing this window will NOT stop the training." -ForegroundColor DarkGray

    if (-not $pipeline) {
        Write-Host "Pipeline has stopped or completed. Press Enter to close."
        Read-Host | Out-Null
        break
    }
    Start-Sleep -Seconds 2
}
